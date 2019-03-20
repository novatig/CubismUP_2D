//
//  Shape.h
//  CubismUP_2D
//
//  Virtual shape class which defines the interface
//  Default simple geometries are also provided and can be used as references
//
//  This class only contains static information (position, orientation,...), no dynamics are included (e.g. velocities,...)
//
//  Created by Christian Conti on 3/6/15.
//  Copyright (c) 2015 ETHZ. All rights reserved.
//

#pragma once

#include "../Shape.h"

class Glider : public Shape
{
 public:
  const int rewardType; /* 1=time 2=energy */
  const Real semiAxis[2];
  //Characteristic scales:
  const Real majax = std::max(semiAxis[0], semiAxis[1]);
  const Real minax = std::min(semiAxis[0], semiAxis[1]);
  const Real velscale = std::sqrt((rhoS-1)*std::fabs(sim.gravity[1])*minax); // 0.01566045976
  const Real lengthscale = majax, timescale = majax/velscale;
  const Real DTactions_nonDim = 0.5;
  const Real DTactions = DTactions_nonDim * timescale;
  const Real beta = minax/majax;
  //const Real torquescale = M_PI/8*std::pow(majax*(1-beta*beta)*velscale,2)/beta;
  // pi/8*((0.125^2-0.025^2)*0.01566045976)^2 /0.125/0.025 = 0.000006934280382
  const Real torquescale = M_PI*majax*majax*velscale*velscale;
  // pi*0.125^2*0.01566045976^2 = 0.00001203868122
  const Real termRew_fac = rewardType==1 ? 50 : 10;
  double old_angle = 0;
  double old_torque = 0;
  double old_distance = 100;
  double appliedTorque = 0;
  double energy = 0, energySurf = 0;

  void resetAll() override
  {
    old_angle = 0;
    old_torque = 0;
    old_distance = 100;
    appliedTorque = 0;
    energy = 0;
    energySurf = 0;
    Shape::resetAll();
  }

  void act(std::vector<double> action);
  std::vector<double> state() const;

  double reward();
  double terminalReward();

  bool isOver();

  Glider(SimulationData& s, ArgumentParser& p, double C[2] );

  Real getCharLength() const override
  {
    return 2 * majax;
  }
  Real getCharMass() const override { return M_PI * semiAxis[0] * semiAxis[1]; }

  void outputSettings(std::ostream &outStream) const override
  {
    outStream << "Glider\n";
    outStream << "axes " << semiAxis[0] <<" "<< semiAxis[1] << std::endl;

    Shape::outputSettings(outStream);
  }

  void create(const std::vector<BlockInfo>& vInfo) override;
  void updateVelocity(double dt) override;
  void computeForces() override;
};
