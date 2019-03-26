//
//  CubismUP_2D
//  Copyright (c) 2018 CSE-Lab, ETH Zurich, Switzerland.
//  Distributed under the terms of the MIT license.
//
//  Created by Guido Novati (novatig@ethz.ch).
//

#pragma once

#include "../Shape.h"
#define SMART_ELLIPSE
#define SMART_AR 0.2

class SmartCylinder : public Shape
{
  const double radius;
 public:
  Real energy = 0, energySurf = 0;

  void act(std::vector<double> action, const Real velScale);

  std::vector<double> state(const Real OX, const Real OY, const Real velScale) const;

  double reward(const Real velScale);

  SmartCylinder(SimulationData& s, cubism::ArgumentParser& p, double C[2] ) :
  Shape(s,p,C), radius( p("-radius").asDouble(0.1) ) {
    printf("Created a SmartCylinder with: R:%f rho:%f\n",radius, rhoS);
  }

  Real getCharLength() const override
  {
    return 2 * radius;
  }
  #ifdef SMART_ELLIPSE
  Real getCharMass() const override { return M_PI * radius*radius * SMART_AR; }
  #else
  Real getCharMass() const override { return M_PI * radius*radius; }
  #endif

  void outputSettings(std::ostream &outStream) const override
  {
    outStream << "Disk\n";
    outStream << "radius " << radius << std::endl;

    Shape::outputSettings(outStream);
  }

  void create(const std::vector<cubism::BlockInfo>& vInfo) override;
  void updatePosition(double dt) override;
  void updateVelocity(double dt) override;
  void computeForces() override;
};
