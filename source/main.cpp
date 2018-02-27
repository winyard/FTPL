//#include "Skyrme.hpp"
//#include "BabySkyrme.hpp"
//#include "SkyrmeWithMeson.hpp"
//#include "Vorticies.hpp"
//#include "CompactVorticieswMetric.hpp"
#include "CompactVorticies.hpp"

int main(int argc, char * argv[]) {

    double c1 = 0.141;
    double c2 = 0.198;

    int choice = -29;

    Eigen::initParallel();

    if(choice == -29){
        int Npoints = 100;
        FTPL::VortexModel my_model(Npoints, Npoints, false);
        cout << "created!\n";
        Eigen::Vector2d input(1.0, 0);
        my_model.phi1->fill(input);
        my_model.phi2->fill(input);
        input[0] = 0.0;
        my_model.A->fill(input);

        double space = 0.01;
        my_model.setSpacing({7.0*space, 7.0*space});

        cout << "setting anisoptropies\n";

        int N = 2;
        int outp = 0;

        double l1 = 2.0;
        double l2 = 2.0;
        double m1 = 1.0;
        double m2 = 1.0;
        double k = 0.0;
        double l1x = 4.5;
        double l1y = 0.7;
        double l1xy = 0.0;
        double l2x = 1.0;
        double l2y = 1.0;
        double l2xy = 0.0;

        vector<vector<Eigen::Matrix2d>> Q;

        Q.resize(2);
        Q[0].resize(2);
        Q[1].resize(2);

        Q[0][0](0,0) = l1x;
        Q[0][0](1,1) = l1y;
        Q[0][0](1,0) = l1xy;
        Q[0][0](0,1) = l1xy;

        Q[1][1](0,0) = l2x;
        Q[1][1](1,1) = l2y;
        Q[1][1](1,0) = l2xy;
        Q[1][1](0,1) = l2xy;

        Q[0][1](0,0) = 0.0;
        Q[0][1](1,1) = 0.0;
        Q[0][1](1,0) = 0.0;
        Q[0][1](0,1) = 0.0;

        Q[1][0](0,0) = 0.0;
        Q[1][0](1,1) = 0.0;
        Q[1][0](1,0) = 0.0;
        Q[1][0](0,1) = 0.0;

        my_model.setParameters(l1,l2,m1,m2,k);
        my_model.setAnisotropy(Q);

        cout << "initial conditions\n";

        my_model.setTimeInterval(0.005);
        my_model.initialCondition(N, 0, 0, 0.0 );

        cout << "now clac energy\n";
        my_model.updateEnergy();
        cout << "Energy = " << my_model.getEnergy() << "\n";
        my_model.updateCharge();
        cout << "Charge = " << my_model.getCharge() << "\n";

        if(outp==1) {
            my_model.plotEnergy();
            my_model.plotCharge();
        }

/*
        for(int i = 0; i < 100; i ++){
            for(int j = 0; j < 100; j ++){
                vector<int> pos(2);
                pos[0] = i;
                pos[1] = j;
                double xmax = 100*space/2.0;
                double ymax = 100*space/2.0;
                double x = i * space - xmax;
                double y = j * space - ymax;
                if(y != 0) {
                    double func = atan(x / y);
                    double dxfunc = y / (x * x + y * y);
                    double dyfunc = -x / (x * x + y * y);
                    cout << func << "," << dxfunc <<"," << dyfunc << "\n";
                    Eigen::Vector2d val(dxfunc, dyfunc);
                    Eigen::Matrix2d rot;
                    rot << cos(func), -sin(func), sin(func), cos(func);
                    my_model.phi1->setData(rot*my_model.phi1->getData(pos),pos);
                    my_model.phi2->setData(rot*my_model.phi2->getData(pos),pos);
                    my_model.A->setData(my_model.A->getData(pos)+val,pos);
                }
            }
        }
        cout << "GAUGE TRANSFORMATION PERFORMED!\n";
        cout << "now clac energy\n";
        my_model.updateEnergy();
        cout << "Energy = " << my_model.getEnergy() << "\n";
        my_model.updateCharge();
        cout << "Charge = " << my_model.getCharge() << "\n";

        my_model.plotPhi1sq();
        my_model.plotPhi2sq();
        my_model.plotAx();
        my_model.plotAy();
        */
        /*int I = Npoints;
        for(int i = 0; i < I; i++){
            double windt = N*2.0*M_PI*i/I;
            double windA = N*2.0*M_PI/(space*I);
            vector<int> posv(2);
            Eigen::Vector2d base(1,0);
            Eigen::Vector2d baseA(0,0);
            Eigen::Vector2d addA(0,windA);
            Eigen::Matrix2d rotation;
            rotation << cos(windt), -sin(windt), sin(windt), cos(windt);
            posv[1] = I-1;
            posv[0] = i;
            double pos = I*posv[0] + posv[1];
            my_model.phi1->data[pos] = base;
            my_model.phi2->data[pos]  = base;
            my_model.A->data[pos]  = baseA;
            posv[1] = I-2;
            pos = I*posv[0] + posv[1];
            my_model.phi1->data[pos]  = base;
            my_model.phi2->data[pos]  = base;
            my_model.A->data[pos]  = baseA;
            posv[1] = 0;
            pos = I*posv[0] + posv[1];
            my_model.phi1->data[pos] = base;
            my_model.phi2->data[pos] = base;
            my_model.A->data[pos] = baseA;
            posv[1] = 1;
            pos = I*posv[0] + posv[1];
            my_model.phi1->data[pos] = base;
            my_model.phi2->data[pos] = base;
            my_model.A->data[pos] = baseA;
        }
        for(int i = 0; i < I; i++){
            double windt = N*2.0*M_PI*i/I;
            double windA = N*2.0*M_PI/(space*I);
            vector<int> posv(2);
            Eigen::Vector2d base(1,0);
            Eigen::Vector2d baseA(0,0);
            Eigen::Vector2d addA(0,windA);
            Eigen::Matrix2d rotation;
            rotation << cos(windt), -sin(windt), sin(windt), cos(windt);
            posv[0] = I-1;
            posv[1] = i;
            double pos = I*posv[0] + posv[1];
            my_model.phi1->data[pos] = rotation*base;
            my_model.phi2->data[pos] = rotation*base;
            my_model.A->data[pos] = baseA + addA;
            posv[0] = I-2;
            pos = I*posv[0] + posv[1];
            my_model.phi1->data[pos] = rotation*base;
            my_model.phi2->data[pos] = rotation*base;
            my_model.A->data[pos] = baseA + addA;
            posv[0] = 0;
            pos = I*posv[0] + posv[1];
            my_model.phi1->data[pos] = base;
            my_model.phi2->data[pos] = base;
            my_model.A->data[pos] = baseA;
            posv[0] = 1;
            pos = I*posv[0] + posv[1];
            my_model.phi1->data[pos] = base;
            my_model.phi2->data[pos] = base;
            my_model.A->data[pos] = baseA;
        }

        my_model.gradientFlow(5000, 250);*/


        my_model.updateEnergy();
        cout << "final Energy = " << my_model.getEnergy() << "\n";
        my_model.updateCharge();
        cout << "final Charge = " << my_model.getCharge() << "\n";

            my_model.storeFields();
            my_model.plotEnergy();
            my_model.plotCharge();

        my_model.setSpacing({space, space});

        //my_model.gradientFlow(5000, 250);
/*
        my_model.updateEnergy();
        cout << "final Energy = " << my_model.getEnergy() << "\n";
        my_model.updateCharge();
        cout << "final Charge = " << my_model.getCharge() << "\n";

        my_model.plotEnergy();
        my_model.plotCharge();*/

        //now gauge out the x-component of the A


        my_model.makePeriodic();
        my_model.phi1->boundaryconstant = N;
        my_model.phi2->boundaryconstant = N;
        my_model.A->boundaryconstant = N;


        cout << "now clac energy\n";
        my_model.updateEnergy();
        cout << "Energy = " << my_model.getEnergy() << "\n";
        my_model.updateCharge();
        cout << "Charge = " << my_model.getCharge() << "\n";

        if(outp==1) {
            my_model.plotEnergy();
            my_model.plotCharge();
        }

        my_model.setTimeInterval(0.01 * space * space);

        Timer tmr;
        for(int i =0; i < 100; i++) {
            int plotornot = 1;

            my_model.gradientFlow(5000, 1000);
            my_model.setTimeInterval(0.25 * space * space);

            my_model.storeFields();

            if(i == 0){my_model.perturbL();}

            my_model.updateEnergy();
            cout << "final Energy = " << my_model.getEnergy() << "\n";
            my_model.updateCharge();
            cout << "final Charge = " << my_model.getCharge() << "\n";
            my_model.outputSpace();
            if(plotornot==1) {
                //my_model.plotEnergy();
                //my_model.plotCharge();
            }
        }


        cout << "ALL DONE!!\n";

        my_model.plotEnergy();
        my_model.plotCharge();


    }


}
/*
    if(choice == - 19){
        FTPL::VortexModel my_model(100, 100, false);

        double space = 0.1;
        my_model.setSpacing({space, space});

        my_model.setParameters(50.0,50.0,0.1,1.0,0.0);
        my_model.setAnisotropy(1,1,1,1);
        my_model.setCouplings(1.0,3.0);

        my_model.initialCondition(1, 0, 0, 0);

        //my_model.addSoliton(1, 0.5, 0.0, 0);

        cout << "now clac energy\n";
        my_model.updateEnergy();
        cout << "Energy = " << my_model.getEnergy() << "\n";
        my_model.updateCharge();
        cout << "Charge = " << my_model.getCharge() << "\n";

            my_model.plotEnergy();
            my_model.plotCharge();

        my_model.setTimeInterval(0.01 * space * space);

        Timer tmr;
        for(int i =0; i < 5; i++) {

            my_model.gradientFlow(4000, 2000);
            my_model.setTimeInterval(0.25 * space * space);

            cout << "now clac energy\n";
            my_model.updateEnergy();
            cout << "Energy = " << my_model.getEnergy() << "\n";
            my_model.updateCharge();
            cout << "Charge = " << my_model.getCharge() << "\n";

            my_model.plotEnergy();
            my_model.plotCharge();

        }


        cout << "ALL DONE!!\n";

        my_model.plotEnergy();
        my_model.plotCharge();





    }}
    /*

    if(choice == -10){
            FTPL::VortexModel my_model(100, 100, false);
            cout << "created!\n";
            Eigen::Vector2d input(1.0, 0);
            my_model.phi1->fill(input);
            my_model.phi2->fill(input);
            input[0] = 0.0;
            my_model.A->fill(input);

            double space = 0.2;
            my_model.setSpacing({space, space});

            int N = 2;
            int outp = 1;

            double l1 = 4.0;
            double l2 = 4.0;
            double m1 = 1.0;
            double m2 = 1.0;
            double k = 0.5;
            double l1x = 6.0;
            double l1y = 0.5;
            double l1xy = 0.0;
            double l2x = 1.0;
            double l2y = 1.0;
            double l2xy = 0.0;

            l1 = 2.0;
            l2 = 2.0;
            m1 = 1.0;
            m2 = 1.0;
            k = 0.5;
            l1x = 5.0;
            l1y = 0.5;
            l2x = 1.0;
            l2y = 1.0;

            my_model.setParameters(l1,l2,m1,m2,k);
            my_model.setAnisotropy(sqrt(l1x),sqrt(l1y),sqrt(l1xy),sqrt(l2x),sqrt(l2y),sqrt(l2xy));
            {
                ofstream outparam("/home/tom/parameters");
                outparam << l1 << " " << l2 << "\n";
                outparam << l1x << " " << l1y << " " << l1xy <<" " << l2x << " " << l2y << " " << l2xy <<"\n";
                outparam << k << "\n";
                outparam << N << "\n";
                outparam << space << "\n";
                outparam << outp << "\n";
            }
            cout << "All set to now change bounary conditions\n";




            my_model.setTimeInterval(0.005);
            cout << "do initial conditions\n";
            my_model.initialCondition(N, 0, 0, 0);

            for(int i=0;i<N;i++) {

                // my_model.addSoliton(-1, 0.45 * space*100, 100*(-0.5*space + (i+1)*space/(N+2)),0);
            }
            //my_model.addSoliton(5,0.15*space*100,0.15*space*100,0);
            //my_model.addSoliton(5,0.2*space*100,-0.145*space*100,0);
            //my_model.addSoliton(5,-0.1*space*100,0.156754*space*100,0);
            //my_model.addSoliton(5,-0.15*space*100,-0.15*space*100,0);

            my_model.makePeriodic();
            my_model.phi1->boundaryconstant = N;
            my_model.phi2->boundaryconstant = N;
            my_model.A->boundaryconstant = N;


            cout << "now clac energy\n";
            my_model.updateEnergy();
            cout << "Energy = " << my_model.getEnergy() << "\n";
            my_model.updateCharge();
            cout << "Charge = " << my_model.getCharge() << "\n";

            if(outp==1) {
                my_model.plotEnergy();
                my_model.plotCharge();
            }


            my_model.setTimeInterval(0.01 * space * space);

            Timer tmr;
            for(int i =0; i < 1000000000; i++) {
                int plotornot = 1;

                my_model.gradientFlow(10000, 2000);
                my_model.setTimeInterval(0.25 * space * space);
                double lambda1, lambda2, lambdax1, lambdaxy1, lambdax2, lambday1, lambday2, lambdaxy2, k0, winding;
                {
                    ifstream infile("/home/tom/parameters");
                    infile >> lambda1 >> lambda2;
                    infile >> lambdax1 >> lambday1 >> lambdaxy1 >> lambdax2 >> lambday2 >> lambdaxy2;
                    infile >> k0;
                    infile >> winding;
                    infile >> space;
                    infile >> plotornot;
                }

                my_model.updateEnergy();
                cout << "final Energy = " << my_model.getEnergy() << "\n";
                my_model.updateCharge();
                cout << "final Charge = " << my_model.getCharge() << "\n";
                if(plotornot==1) {
                    my_model.plotEnergy();
                    my_model.plotCharge();
                }
                my_model.setParameters(lambda1,lambda2,1.0,1.0,k0);
                my_model.setAnisotropy(sqrt(lambdax1),sqrt(lambday1),sqrt(lambdaxy1),sqrt(lambdax2),sqrt(lambday2),sqrt(lambdaxy2));
                my_model.phi1->boundaryconstant = winding;
                my_model.phi2->boundaryconstant = winding;
                my_model.A->boundaryconstant = winding;
                my_model.setSpacing({space, space});

            }


            cout << "ALL DONE!!\n";

            my_model.plotEnergy();
            my_model.plotCharge();


    }

    if(choice == -8) {
        FTPL::BabySkyrmeModel my_model(100, 100);

        Eigen::Vector3d input(0, 0, 1);
        Eigen::Vector3d dtinput(0, 0, 0);
        my_model.f->fill(input);

        double space = 4.0/99;
        my_model.setSpacing({space, space});
        my_model.setParameters(1.0, sqrt(0.1));

        my_model.setTimeInterval(1.0 * space);

        my_model.addSoliton(1, 0, 0.0, 0);
        //my_model.addSoliton(1, -2.1, 0, M_PI);
        //my_model.addSoliton(1, -1, 0, 3.12);


        my_model.updateEnergy();
        cout << "Initial energy = " << my_model.getEnergy() << "\n";
        my_model.updateCharge();
        cout << "Initial charge = " << my_model.getCharge() << "\n";
        cout << "now make periodic\n";

        my_model.setAllBoundaryType({1,1,1,1});
        my_model.f->setboundarytype({1,1,1,1});
        my_model.setbdw({0,0,0,0});



        my_model.updateEnergy();
        cout << "Initial energy = " << my_model.getEnergy() << "\n";
        my_model.updateCharge();
        cout << "Initial charge = " << my_model.getCharge() << "\n";

        my_model.plotEnergy();

        my_model.setTimeInterval(0.1 * space * space);
        for(int i = 0; i < 150; i++) {
            if(i==25){my_model.setTimeInterval(0.1 * space * space);}
            my_model.gradientFlow(1000, 10000);
            my_model.updateEnergy();
            cout << "Final final Energy = " << my_model.getEnergy() << ", normalised is "
                 << my_model.getEnergy() / (4 * M_PI) << "\n";
            my_model.updateCharge();
            cout << "Final final Charge = " << my_model.getCharge() << "\n";
            cout << "current metric :\n";
            cout << my_model.g(0,0) << " " << my_model.g(0,1) << "\n";
            cout << my_model.g(1,0) << " " << my_model.g(1,1) << "\n";

        }
        cout << "ALL DONE!!\n";

        my_model.plotEnergy();

    }

    if(choice == -6) {
        FTPL::BabySkyrmeModel my_model(100, 100);

        Eigen::Vector3d input(0, 0, 1);
        Eigen::Vector3d dtinput(0, 0, 0);
        my_model.f->fill(input);

        double space = 8.0/99;
        my_model.setSpacing({space, space});
        my_model.setParameters(1.0, sqrt(0.1));

        my_model.setTimeInterval(1.0 * space);

        my_model.addSoliton(1, 2.1, 0.0, 0);
        my_model.addSoliton(1, -2.1, 0, M_PI);
        //my_model.addSoliton(1, -1, 0, 3.12);


        my_model.updateEnergy();
        cout << "Initial energy = " << my_model.getEnergy() << "\n";
        my_model.updateCharge();
        cout << "Initial charge = " << my_model.getCharge() << "\n";
        cout << "now make periodic\n";

        my_model.setAllBoundaryType({1,1,1,1});
        my_model.f->setboundarytype({1,1,1,1});
        my_model.setbdw({0,0,0,0});



        my_model.updateEnergy();
        cout << "Initial energy = " << my_model.getEnergy() << "\n";
        my_model.updateCharge();
        cout << "Initial charge = " << my_model.getCharge() << "\n";

        my_model.plotEnergy();

        my_model.setTimeInterval(0.1 * space * space);
        my_model.gradientFlow(10000, 1000);
        my_model.updateEnergy();
        cout << "Final final Energy = " << my_model.getEnergy() << ", normalised is " << my_model.getEnergy()/(4*M_PI) << "\n";
        my_model.updateCharge();
        cout << "Final final Charge = " << my_model.getCharge() << "\n";
        cout << "ALL DONE!!\n";

        my_model.plotEnergy();

    }

    if(choice == -5){
            FTPL::VortexModel my_model(100, 100, false);
            cout << "created!\n";
            Eigen::Vector2d input(1.0, 0);
            my_model.phi1->fill(input);
            my_model.phi2->fill(input);
            input[0] = 0.0;
            my_model.A->fill(input);

            double space = 0.2;
            my_model.setSpacing({space, space});

            int N = 20;
            int outp = 1;

            double l1 = 4.0;
            double l2 = 4.0;
            double m1 = 1.0;
            double m2 = 1.0;
            double k = 0.5;
            double l1x = 6.0;
            double l1y = 0.5;
            double l2x = 1.0;
            double l2y = 1.0;

            l1 = 2.0;
            l2 = 2.0;
            m1 = 1.0;
            m2 = 1.0;
            k = 0.5;
            l1x = 5.0;
            l1y = 0.5;
            l2x = 1.0;
            l2y = 1.0;

            my_model.setParameters(l1,l2,m1,m2,k);
            //my_model.setAnisotropy(sqrt(l1x),sqrt(l1y),sqrt(l2x),sqrt(l2y));
        {
            ofstream outparam("/home/tom/parameters");
            outparam << l1 << " " << l2 << "\n";
            outparam << l1x << " " << l1y << " " << l2x << " " << l2y << "\n";
            outparam << k << "\n";
            outparam << N << "\n";
            outparam << space << "\n";
            outparam << outp << "\n";
        }
            cout << "All set to now change bounary conditions\n";




            my_model.setTimeInterval(0.005);
            cout << "do initial conditions\n";
            my_model.initialCondition(N, 0, 0, 0);

        for(int i=0;i<N;i++) {

           // my_model.addSoliton(-1, 0.45 * space*100, 100*(-0.5*space + (i+1)*space/(N+2)),0);
        }
        my_model.addSoliton(5,0.15*space*100,0.15*space*100,0);
        my_model.addSoliton(5,0.2*space*100,-0.145*space*100,0);
        my_model.addSoliton(5,-0.1*space*100,0.156754*space*100,0);
        my_model.addSoliton(5,-0.15*space*100,-0.15*space*100,0);

        my_model.makePeriodic();
        my_model.phi1->boundaryconstant = N;
        my_model.phi2->boundaryconstant = N;
        my_model.A->boundaryconstant = N;


            cout << "now clac energy\n";
            my_model.updateEnergy();
            cout << "Energy = " << my_model.getEnergy() << "\n";
            my_model.updateCharge();
            cout << "Charge = " << my_model.getCharge() << "\n";

        if(outp==1) {
            my_model.plotEnergy();
            my_model.plotCharge();
        }


        my_model.setTimeInterval(0.01 * space * space);

            Timer tmr;
            for(int i =0; i < 1000000000; i++) {
                int plotornot = 1;

                my_model.gradientFlow(10000, 2000);
                my_model.setTimeInterval(0.25 * space * space);
                double lambda1, lambda2, lambdax1, lambdax2, lambday1, lambday2, k0, winding;
                {
                    ifstream infile("/home/tom/parameters");
                    infile >> lambda1 >> lambda2;
                    infile >> lambdax1 >> lambday1 >> lambdax2 >> lambday2;
                    infile >> k0;
                    infile >> winding;
                    infile >> space;
                    infile >> plotornot;
                }

                my_model.updateEnergy();
                cout << "final Energy = " << my_model.getEnergy() << "\n";
                my_model.updateCharge();
                cout << "final Charge = " << my_model.getCharge() << "\n";
                if(plotornot==1) {
                    my_model.plotEnergy();
                    my_model.plotCharge();
                }
                    my_model.setParameters(lambda1,lambda2,1.0,1.0,k0);
                   // my_model.setAnisotropy(sqrt(lambdax1),sqrt(lambday1),sqrt(lambdax2),sqrt(lambday2));
                    my_model.phi1->boundaryconstant = winding;
                    my_model.phi2->boundaryconstant = winding;
                    my_model.A->boundaryconstant = winding;
                    my_model.setSpacing({space, space});

            }


            cout << "ALL DONE!!\n";

            my_model.plotEnergy();
            my_model.plotCharge();


        }

        if(choice == -2){
            MPI::Init(argc, argv);
            printMPIDetails();
            FTPL::BabySkyrmeModel my_model(100, 100, true);

            Eigen::Vector3d input(0, 0, 1);
            Eigen::Vector3d dtinput(0, 0, 0);
            my_model.f->fill(input);
            my_model.f->fill_dt(dtinput);

            double space = 0.2;
            my_model.setSpacing({space, space});
            my_model.setParameters(1.0, sqrt(0.1));
            my_model.setAllBoundaryType({1,1,1,1});
            my_model.setbdw({0,0,0,0});

            my_model.setTimeInterval(0.5 * space);

            my_model.addSoliton(1, 1, 0, 0);
            //my_model.addSoliton(1,-1,0,0.24);
            //my_model.addSoliton(1, -1, 0, 3.12);

            my_model.updateEnergy();
            cout << "Initial energy = " << my_model.getEnergy() << "\n";
            my_model.updateCharge();
            cout << "Initial charge = " << my_model.getCharge() << "\n";

            //my_model.plotEnergy();

            Timer tmr;
            my_model.RK4(2000,true,10);
            cout << "250 loops finished in " << tmr.elapsed() << " sec\n";
            my_model.updateEnergy();
            cout << "Final Energy = " << my_model.getEnergy() << "\n";
            my_model.updateCharge();
            cout << "Final Charge = " << my_model.getCharge() << "\n";
            //my_model.plotEnergy();
            my_model.setTimeInterval(0.5 * space * space * space);
            my_model.gradientFlow(100, 10000);
            my_model.updateEnergy();
            cout << "Final final Energy = " << my_model.getEnergy() << "\n";
            my_model.updateCharge();
            cout << "Final final Charge = " << my_model.getCharge() << "\n";
            cout << "ALL DONE!!\n";




        }

        if(choice == -1){
            MPI::Init(argc, argv);
            printMPIDetails();
            FTPL::SkyrmeModelwithMeson my_model(10,10,10,false);
            my_model.addVectorMeson();
            my_model.load("in_field", true);
            //my_model.initialCondition(1,0,0,0,0);
            my_model.updateEnergy();
            my_model.updateCharge();
            cout << "the loaded model has energy = " << my_model.getEnergy() << " and charge = " << my_model.getCharge() << "\n";
            my_model.setTimeInterval(0.005);
            Timer tmr;
            my_model.MPIAnnealing(10,1,1,100,2);
            cout << "10 loops finished in " << tmr.elapsed() << " sec\n";
            my_model.save("temp_field");
        }

        if(choice == 0){
            FTPL::SkyrmeModelwithMeson my_model(10,10,10,false);
            my_model.addVectorMeson();
            my_model.load("temp_field", false);
            //my_model.initialCondition(1,0,0,0,0);
            my_model.updateEnergy();
            my_model.updateCharge();
            cout << "the loaded model has energy = " << my_model.getEnergy() << " and charge = " << my_model.getCharge() << "\n";
            my_model.printParameters();
            my_model.setTimeInterval(0.005);
            cout << "vector meson added\n";
            my_model.updateEnergy();
            my_model.updateCharge();
            cout << "the loaded model with added vector meson has energy = " << my_model.getEnergy() << " and charge = " << my_model.getCharge() << "\n";
            Timer tmr;
            my_model.annealing(10000, 1000000, 2);
            cout << "10 loops finished in " << tmr.elapsed() << " sec\n";
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

            my_model.setTimeInterval(0.5 * space);
            my_model.initialCondition(1, 0, 0, 0, 0);

            my_model.updateEnergy();
            cout << "Energy = " << my_model.getEnergy() << "\n";
            my_model.updateCharge();
            cout << "Charge = " << my_model.getCharge() << "\n";

            cout << "plotting energy\n";
            my_model.plotEnergy();
            cout << "GOGOGO!\n";
            Timer tmr;
            my_model.RK4(200, true, 1);
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

        if(choice == 4) {
            FTPL::BabySkyrmeModel my_model(100, 100, true);

            Eigen::Vector3d input(0, 0, 1);
            Eigen::Vector3d dtinput(0, 0, 0);
            my_model.f->fill(input);
            my_model.f->fill_dt(dtinput);

            double space = 0.2;
            my_model.setSpacing({space, space});
            my_model.setParameters(1.0, sqrt(0.1));

            my_model.setTimeInterval(0.5 * space);

            my_model.addSoliton(1, 1, 0, 0);
            //my_model.addSoliton(1, -1, 0, 3.12);

            my_model.updateEnergy();
            cout << "Initial energy = " << my_model.getEnergy() << "\n";
            my_model.updateCharge();
            cout << "Initial charge = " << my_model.getCharge() << "\n";

            //my_model.plotEnergy();

            Timer tmr;
            my_model.RK4(2000,true,10);
            cout << "250 loops finished in " << tmr.elapsed() << " sec\n";
            my_model.updateEnergy();
            cout << "Final Energy = " << my_model.getEnergy() << "\n";
            my_model.updateCharge();
            cout << "Final Charge = " << my_model.getCharge() << "\n";
            //my_model.plotEnergy();
            my_model.setTimeInterval(0.5 * space * space * space);
            my_model.gradientFlow(100, 10000);
            my_model.updateEnergy();
            cout << "Final final Energy = " << my_model.getEnergy() << "\n";
            my_model.updateCharge();
            cout << "Final final Charge = " << my_model.getCharge() << "\n";
            cout << "ALL DONE!!\n";
        }

        if(choice == 5) {
            FTPL::BabySkyrmeModel my_model(100, 100, false);
            cout << "created!\n";
            Eigen::Vector3d input(0, 0, 1);
            my_model.f->fill(input);

            double space = 0.2;
            my_model.setSpacing({space, space});
            my_model.setParameters(1.0, sqrt(0.1));

            my_model.setTimeInterval(0.005);
            cout << "do initial conditions\n";
            my_model.initialCondition(1, 0, 0, 0);
            cout << "now clac energy\n";
            my_model.updateEnergy();
            cout << "Energy = " << my_model.getEnergy() << "\n";
            my_model.updateCharge();
            cout << "Charge = " << my_model.getCharge() << "\n";


            cout << "Time to test the anealing method!\n";
            Timer tmr;
            my_model.annealing(2000, 20000, true);
            cout << "20 loops finished in" << tmr.elapsed()/60 << " mins\n";
            my_model.setTimeInterval(0.001);
            my_model.annealing(400, 20000, true);
            cout << "another 20 loops finished in " << tmr.elapsed()/60 << " mins\n";
            my_model.setTimeInterval(0.001);
            my_model.annealing(400, 20000, true);
            cout << "another 20 loops finished in " << tmr.elapsed()/60 << " mins\n";
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
/*
    }

    if(choice == 6){

            FTPL::SkyrmeModel my_model(100, 100, 100, true);

            Eigen::Vector4d input(1, 0, 0, 0);
            my_model.f->fill(input);

            double space = 0.15;
            my_model.setSpacing({space, space, space});
            my_model.setParameters(sqrt(c1)*sqrt(8.0), (1.0/sqrt(c2))*sqrt(0.5), 0.0);

            my_model.setTimeInterval(0.01);
            my_model.initialCondition(1, 0, 0, 0, 0);

            my_model.updateEnergy();
            cout << "Energy = " << my_model.getEnergy() << "\n";
            my_model.updateCharge();
            cout << "Charge = " << my_model.getCharge() << "\n";

        cout << "Time to test the anealing method!\n";
        Timer tmr;
        my_model.annealing(2000, 20000, true);
        cout << "20 loops finished in" << tmr.elapsed()/60 << " mins\n";
        my_model.setTimeInterval(0.001);
        my_model.annealing(400, 20000, true);
        cout << "another 20 loops finished in " << tmr.elapsed()/60 << " mins\n";
        my_model.setTimeInterval(0.001);
        my_model.annealing(400, 20000, true);
        cout << "another 20 loops finished in " << tmr.elapsed()/60 << " mins\n";
        cout << "ALL DONE!!\n";
            my_model.plotEnergy();

    }

    if(choice == 7){

        FTPL::SkyrmeModel my_model(100, 100, 100, true);

        Eigen::Vector4d input(1, 0, 0, 0);
        my_model.f->fill(input);

        double space = 0.15;
        my_model.setSpacing({space, space, space});
        my_model.setParameters(sqrt(c1)*sqrt(8.0), (1.0/sqrt(c2))*sqrt(0.5), 0.0);

        my_model.setTimeInterval(0.01);
        my_model.initialCondition(1, 0, 0, 0, 0);

        my_model.updateEnergy();
        cout << "Energy = " << my_model.getEnergy() << "\n";
        my_model.updateCharge();
        cout << "Charge = " << my_model.getCharge() << "\n";

        cout << "Time to test the anealing method!\n";
        Timer tmr;
        my_model.annealing(2000, 20000, true);
        cout << "20 loops finished in" << tmr.elapsed()/60 << " mins\n";
        my_model.setTimeInterval(0.001);
        my_model.annealing(400, 20000, true);
        cout << "another 20 loops finished in " << tmr.elapsed()/60 << " mins\n";
        my_model.setTimeInterval(0.001);
        my_model.annealing(400, 20000, true);
        cout << "another 20 loops finished in " << tmr.elapsed()/60 << " mins\n";
        cout << "ALL DONE!!\n";
        my_model.plotEnergy();



    }

    if(choice == 8){

        FTPL::BabySkyrmeModel my_model(800, 800, true);

        Eigen::Vector3d input(0, 0, 1);
        Eigen::Vector3d dtinput(0, 0, 0);
        my_model.f->fill(input);
        my_model.f->fill_dt(dtinput);

        double space = 0.05;
        my_model.setSpacing({space, space});
        my_model.setParameters(1.0, sqrt(0.1));

        my_model.setTimeInterval(0.1 * space);
        int tail_size = 4;
        int junc_type = 3;
        int centre_charge = 2;

        if(centre_charge > 0){my_model.initialCondition(centre_charge, 0, 0, 0);}

        cout << "initiating junction type " << junc_type << " with chain lengths " << tail_size << "\n";
        cout << "expected charge = " << centre_charge + tail_size*junc_type << "\n";
        double r = 2.5;
        for(int i = 1; i <= tail_size; i++)
        {
            for(int j = 0; j < junc_type; j++)
            {
                double theta = (2.0*M_PI/junc_type)*j;
                double phase = M_PI*i + j*M_PI/2.0;
                my_model.addSoliton(1, i*r*cos(theta), i*r*sin(theta),phase);
            }
        }
        my_model.f->fill_dt(dtinput);
        my_model.updateEnergy();
        cout << "Initial energy = " << my_model.getEnergy() << "\n";
        my_model.updateCharge();
        cout << "Initial charge = " << my_model.getCharge() << "\n";

        my_model.plot("./outputfileY");

        my_model.save("./savefileY");
        my_model.setTimeInterval(0.5 * space * space * space);
        my_model.gradientFlow(1000, 100);

        my_model.setTimeInterval(0.1 * space);
        my_model.RK4(40,true,10);
        my_model.f->fill_dt(dtinput);
        my_model.RK4(50,true,10);
        my_model.f->fill_dt(dtinput);
        my_model.RK4(60,true,10);
        my_model.f->fill_dt(dtinput);

        my_model.RK4(500,true,10);

        my_model.setTimeInterval(0.5 * space * space * space);
        my_model.gradientFlow(500, 100);

        my_model.setTimeInterval(0.25 * space);
        my_model.RK4(2000,true,10);

        my_model.setTimeInterval(0.5 * space * space * space);
        my_model.gradientFlow(10000, 100);
        //my_model.plotEnergy();
        //my_model.addSoliton(1, -1, 0, 3.12);
        //my_model.plotEnergy();
        my_model.updateEnergy();
        cout << "Initial energy = " << my_model.getEnergy() << "\n";
        my_model.updateCharge();
        cout << "Initial charge = " << my_model.getCharge() << "\n";
        Timer tmr;
        my_model.setTimeInterval(0.5 * space);
        my_model.RK4(2000,true,10);
        cout << "250 loops finished in " << tmr.elapsed() << " sec\n";
        my_model.updateEnergy();
        cout << "Final Energy = " << my_model.getEnergy() << "\n";
        my_model.updateCharge();
        cout << "Final Charge = " << my_model.getCharge() << "\n";
        //my_model.plotEnergy();
        my_model.setTimeInterval(0.5 * space * space * space);
        my_model.gradientFlow(100, 10000);
        my_model.updateEnergy();
        cout << "Final final Energy = " << my_model.getEnergy() << "\n";
        my_model.updateCharge();
        cout << "Final final Charge = " << my_model.getCharge() << "\n";
        cout << "ALL DONE!!\n";



    }

}

*/