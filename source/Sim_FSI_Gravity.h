//
//  Sim_FSI_Gravity.h
//  CubismUP_2D
//
//  Class for the simulation of gravity driven FSI
//
//  Created by Christian Conti on 1/26/15.
//  Copyright (c) 2015 ETHZ. All rights reserved.
//

#pragma once

#include "Simulation_FSI.h"

class Sim_FSI_Gravity : public Simulation_FSI
{
 protected:

  void _diagnostics();

  // should this stuff be moved? - serialize method will do that
  //void _dumpSettings(ostream& outStream);

public:
  Sim_FSI_Gravity(int argc, char ** argv);
  ~Sim_FSI_Gravity();

  void init() override;

  double calcMaxTimestep() override;

  bool advance(const double DT) override;

  #if 0
  Real uOld = 0, vOld = 0;
  double maxA = 0;
  void testUnifRhoDisk()
  {
    // this test only works for constant density disks as it is written now
    const double accMy = (uBody[1]-vOld)/dt;
    const double accMx = (uBody[0]-uOld)/dt;
    vOld = uBody[1];
    uOld = uBody[0];
    const double accT = (shape->getMinRhoS()-1)/(shape->getMinRhoS()+1)*gravity[1];
    const double accN = (shape->getMinRhoS()-1)/(shape->getMinRhoS()  )*gravity[1];
    //if (verbose)
    cout<<"Acceleration with added mass (measured x,y, expected, no added mass)\t"
        <<accMx<<"\t"<<accMy<<"\t"<<accT<<"\t"<<accN<<endl;
    stringstream ss;
    ss<<path2file<<"_addedmass.dat";
    ofstream myfile(ss.str(), fstream::app);
    myfile<<step<<" "<<time<<" "<<accMx<<" "<<accMy<<" "<<accT<<" "<<accN<<endl;
  }
  #endif
};
