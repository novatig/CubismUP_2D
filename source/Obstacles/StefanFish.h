//
//  CubismUP_2D
//  Copyright (c) 2018 CSE-Lab, ETH Zurich, Switzerland.
//  Distributed under the terms of the MIT license.
//
//  Created by Guido Novati (novatig@ethz.ch).
//

#pragma once
#include "Fish.h"

#define STEFANS_SENSORS_STATE

class StefanFish: public Fish
{
  // std::array<Real ,6> curvature_points;
  // std::array<Real ,6> curvature_values;
  // std::array<Real ,6> baseline_points;
  // std::array<Real ,6> baseline_values;
  // Real tau;
  double adjTh = 0, adjDy = 0;
  const Real followX, followY;
  const bool bCorrectTrajectory;
  Real lastTime = 0;
 public:
  mutable double lastTact = 0;
  mutable double lastCurv = 0;
  mutable double oldrCurv = 0;
  void act(const Real lTact, const std::vector<double>& a) const;
  double getLearnTPeriod() const;
  double getPhase(const double t) const;

  void resetAll() override;
  StefanFish(SimulationData&s, cubism::ArgumentParser&p, double C[2]);
  void create(const std::vector<cubism::BlockInfo>& vInfo) override;

  // member functions for state/reward
  std::vector<double> state(Shape*const p) const;
  double reward() const;

};
