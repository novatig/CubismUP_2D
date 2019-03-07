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

#include "Shape.h"
#define SMART_ELLIPSE
#define SMART_AR 0.2

class SmartCylinder : public Shape
{
  const double radius;
 public:
  Real appliedForceX = 0;
  Real appliedForceY = 0;
  Real appliedTorque = 0;
  Real energy = 0, energySurf = 0;

  void act(std::vector<double> action, const Real velScale);

  std::vector<double> state(const Real OX, const Real OY, const Real velScale) const;

  double reward(const Real velScale);

  SmartCylinder(SimulationData& s, ArgumentParser& p, double C[2] ) :
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

  void outputSettings(ostream &outStream) const override
  {
    outStream << "Disk\n";
    outStream << "radius " << radius << endl;

    Shape::outputSettings(outStream);
  }

  void create(const vector<BlockInfo>& vInfo) override;
  void updatePosition(double dt) override;
  void computeVelocities() override;
  void computeForces() override;
};
