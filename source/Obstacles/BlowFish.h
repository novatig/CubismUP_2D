//
//  CubismUP_2D
//  Copyright (c) 2018 CSE-Lab, ETH Zurich, Switzerland.
//  Distributed under the terms of the MIT license.
//
//  Created by Guido Novati (novatig@ethz.ch).
//


#pragma once
#include "../Shape.h"

class BlowFish : public Shape
{
 public:
  Real flapAng_R = 0, flapAng_L = 0;
  Real flapVel_R = 0, flapVel_L = 0;
  Real flapAcc_R = 0, flapAcc_L = 0;

  void resetAll() override;
  const double radius;
  const double rhoTop = 1.5; //top half
  const double rhoBot = 0.5; //bot half
  const double rhoFin = 1.0; //fins

  const Real finLength = 0.5*radius; //fins
  const Real finWidth  = 0.1*radius; //fins
  const Real finAngle0 = M_PI/6; //fins

  const Real attachDist = radius + std::max(finWidth, (Real)sim.getH()*2);
  //Real powerOutput = 0, old_powerOutput = 0;
  const Real deltaRho = 0.5; //ASSUME RHO FLUID == 1
  // analytical period of oscillation for small angles
  const Real timescale = sqrt(3*M_PI*radius/deltaRho/fabs(sim.gravity[1])/8);
  const Real minRho = std::min(rhoTop,rhoBot), maxRho = std::max(rhoTop,rhoBot);

  BlowFish(SimulationData&s, cubism::ArgumentParser&p, double C[2]);

  void updatePosition(double dt) override;

  void create(const std::vector<cubism::BlockInfo>& vInfo) override;

  Real getCharLength() const  override
  {
    return 2 * radius;
  }
  Real getMinRhoS() const override {
    return std::min( {rhoTop, rhoBot, rhoFin} );
  }
  bool bVariableDensity() const override {
    assert(std::fabs(rhoTop-rhoBot)>std::numeric_limits<Real>::epsilon());
    const bool bTop = std::fabs(rhoTop-1.)>std::numeric_limits<Real>::epsilon();
    const bool bBot = std::fabs(rhoBot-1.)>std::numeric_limits<Real>::epsilon();
    return bTop || bBot;
  }

  void outputSettings(std::ostream &outStream) const override
  {
    outStream << "BlowFish\n";
    Shape::outputSettings(outStream);
  }
};
