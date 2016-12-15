#include "Skyrme.hpp"
#include "BabySkyrme.hpp"
#include "SkyrmeWithMeson.hpp"

int main() {

    double c1 = 0.141;
    double c2 = 0.198;

    int choice = 0;

    Eigen::initParallel();

    if(choice == 0){
        FTPL::SkyrmeModelwithMeson my_model(10,10,10,false);
        my_model.addVectorMeson();
        my_model.load("temp_field", false);
        my_model.initialCondition(1,0,0,0,0);
        my_model.updateEnergy();
        my_model.updateCharge();
        cout << "the loaded model has energy = " << my_model.getEnergy() << " and charge = " << my_model.getCharge() << "\n";
        my_model.printParameters();
        my_model.setTimeInterval(0.01);
        cout << "vector meson added\n";
        my_model.updateEnergy();
        my_model.updateCharge();
        cout << "the loaded model with added vector meson has energy = " << my_model.getEnergy() << " and charge = " << my_model.getCharge() << "\n";
        my_model.annealing(10000, 2000000, 1);
        my_model.save("temp_field");
    }

    if(choice == 1) {
        FTPL::SkyrmeModelwithMeson my_model(100, 100, 100, false);
        cout << "created!\n";
        Eigen::Vector4d input(1, 0, 0, 0);
        my_model.f->fill(input);

        double space = 0.15;
        my_model.setSpacing({space, space, space});
        my_model.setParameters(0.0, 0.0, 0.0);
        my_model.printParameters();

        my_model.setTimeInterval(0.008);

        //my_model.addVectorMeson();

        cout << "do initial conditions\n";
        my_model.initialCondition(1, 0, 0, 0, 0);
        cout << "now clac energy\n";
        my_model.updateEnergy();
        cout << "Energy = " << my_model.getEnergy() << "\n";
        my_model.updateCharge();
        cout << "Charge = " << my_model.getCharge() << "\n";


        cout << "Time to test the anealing method!\n";
        Timer tmr;
        my_model.annealing(5, 1000000, 1);
        cout << "10 loops finished in " << tmr.elapsed() << " sec\n";
        my_model.save("temp_field");
    }

    if(choice == 2) {
        FTPL::SkyrmeModel my_model(100, 100, 100, false);
        cout << "created!\n";
        Eigen::Vector4d input(1, 0, 0, 0);
        my_model.f->fill(input);

        double space = 0.15;
        my_model.setSpacing({space, space, space});
        my_model.setParameters(c1*sqrt(8.0), c2*sqrt(0.5), 0.0);

        my_model.setTimeInterval(0.5 * space);
        cout << "do initial conditions\n";
        my_model.initialCondition(1, 0, 0, 0, 0);
        cout << "now clac energy\n";
        my_model.updateEnergy();
        cout << "Energy = " << my_model.getEnergy() << "\n";
        my_model.updateCharge();
        cout << "Charge = " << my_model.getCharge() << "\n";


        cout << "Time to test the anealing method!\n";
        Timer tmr;
        my_model.annealing(10000000, 1000000, 1);
        cout << "10 loops finished in " << tmr.elapsed() << " sec\n";
        cout << "ALL DONE!!\n";
    }

    if(choice == 3) {

        FTPL::SkyrmeModel my_model(100, 100, 100, true);

        Eigen::Vector4d input(1, 0, 0, 0);
        my_model.f->fill(input);

        double space = 0.15;
        my_model.setSpacing({space, space, space});
        my_model.setParameters(sqrt(c1)*sqrt(8.0), (1.0/sqrt(c2))*sqrt(0.5), 0.0);

        my_model.setTimeInterval(0.15 * space);
        my_model.initialCondition(1, 0, 0, 0, 0);

        my_model.updateEnergy();
        cout << "Energy = " << my_model.getEnergy() << "\n";
        my_model.updateCharge();
        cout << "Charge = " << my_model.getCharge() << "\n";

        cout << "plotting energy\n";
        my_model.plotEnergy();
        cout << "GOGOGO!\n";
        Timer tmr;
        my_model.RK4(20, true, 10);
        cout << "1000 loops finished in " << tmr.elapsed() << " sec\n";
        my_model.updateEnergy();
        my_model.updateCharge();
        cout << "charge = " << my_model.getCharge() << "\n";
        my_model.plotEnergy();
        my_model.setTimeInterval(0.5 * space * space * space);
        my_model.gradientFlow(1000, 100);
        cout << "ALL DONE!!\n";
        my_model.updateEnergy();
        my_model.plotEnergy();

    }

   /* FTPL::BabySkyrmeModel my_model(500, 500, true);

    Eigen::Vector3d input(0, 0, 1);
    my_model.f->fill(input);

    double space = 0.1;
    my_model.setSpacing({space, space});
    my_model.setParameters(1.0,sqrt(0.1));

    my_model.setTimeInterval(0.5*space);
    my_model.initialCondition(1, 0, 0, 0);

    my_model.updateEnergy();
    cout << "Energy = " << my_model.getEnergy() << "\n";
    my_model.updateCharge();
    cout << "Charge = " << my_model.getCharge() << "\n";

    my_model.plotEnergy();

    Timer tmr;
    my_model.RK4(100,true,true,1);
    cout << "1000 loops finished in " <<tmr.elapsed() << " sec\n";
    my_model.plotEnergy();
    my_model.setTimeInterval(0.5*space*space*space);
    my_model.gradientFlow(1000, 10, true);
    cout << "ALL DONE!!\n";*/


/*    FTPL::BabySkyrmeModel my_model(100, 100, false);
    cout << "created!\n";
    Eigen::Vector3d input(0, 0, 1);
    my_model.f->fill(input);

    double space = 0.2;
    my_model.setSpacing({space, space});
    my_model.setParameters(1.0,sqrt(0.1));

    my_model.setTimeInterval(0.5*space);
    cout << "do initial conditions\n";
    my_model.initialCondition(1, 0, 0, 0);
    cout << "now clac energy\n";
    my_model.updateEnergy();
    cout << "Energy = " << my_model.getEnergy() << "\n";
    my_model.updateCharge();
    cout << "Charge = " << my_model.getCharge() << "\n";


    cout << "Time to test the anealing method!\n";
    Timer tmr;
    my_model.annealing(100000000,2000000,true);
    cout << "10 loops finished in " <<tmr.elapsed() << " sec\n";
    cout << "ALL DONE!!\n";


    /*cout << "Time to test the RK4 method!\n";
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

