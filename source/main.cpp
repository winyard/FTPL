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
    Eigen::Vector2i pos(7, 19);
    cout << my_model.f->getData(pos) << "\n";
    cout << "the spacing has been set to " << my_model.getSpacing()[0] << " " << my_model.getSpacing()[1] << "\n";
    vector<double> spacing = {0.1, 0.1};
    my_model.setSpacing({0.1, 0.1});
    my_model.setParameters(1.0,sqrt(0.1));
    cout << "the spacing has been set to " << my_model.getSpacing()[0] << " " << my_model.getSpacing()[1] << "\n";
    my_model.initialCondition(1, 0, 0, 0);
    my_model.updateEnergy();
    cout << "Energy = " << my_model.getEnergy() << "\n";
    my_model.updateCharge();
    cout << "Charge = " << my_model.getCharge() << "\n";




    cout << "Running Gradient Flow:\n";
    my_model.setTimeInterval(1.25*0.1*0.1*0.1);

    Timer tmr;
    my_model.gradientFlow(10, 10);
    cout << "100 loops finished in " <<tmr.elapsed() << " sec\n";
}

