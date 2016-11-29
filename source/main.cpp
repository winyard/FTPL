//#include "FieldTheories.hpp"

#include "BabySkyrme.hpp"


#include <memory>
#include <vector>
#include <iostream>
#include <chrono>

/*class Timer
{
public:
    Timer() : beg_(clock_::now()) {}
    void reset() { beg_ = clock_::now(); }
    double elapsed() const {
        return std::chrono::duration_cast<second_>
                (clock_::now() - beg_).count(); }

private:
    typedef std::chrono::high_resolution_clock clock_;
    typedef std::chrono::duration<double, std::ratio<1> > second_;
    std::chrono::time_point<clock_> beg_;
};*/

int main() {

    Eigen::initParallel();

    FTPL::BabySkyrmeModel my_model(200, 200);
    Eigen::Vector3d input(0, 0, 1);
    my_model.f->fill(input);

    double space = 0.2;
    my_model.setSpacing({space, space});
    my_model.setParameters(1.0,sqrt(0.1));
    my_model.setTimeInterval(space*space*space);
    my_model.initialCondition(1, 0, 0, 0);
    my_model.updateEnergy();
    cout << "Energy = " << my_model.getEnergy() << "\n";
    my_model.updateCharge();
    cout << "Charge = " << my_model.getCharge() << "\n";


    cout << "Running Gradient Flow:\n";


    Timer tmr;
    my_model.gradientFlow(100, 10, true);
    cout << "1000 loops finished in " <<tmr.elapsed() << " sec\n";
}

