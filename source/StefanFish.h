//
//  IF2D_StefanFishOperator.h
//  IF2D_ROCKS
//
//  Created by Wim van Rees on 25/11/14.
//
//

#pragma once
#include "Fish.h"
class StefanFish: public Fish
{
  // std::array<Real ,6> curvature_points;
  // std::array<Real ,6> curvature_values;
  // std::array<Real ,6> baseline_points;
  // std::array<Real ,6> baseline_values;
  // Real tau;
  Real adjTh = 0, adjDy = 0;
  const Real followX, followY;
  const bool bCorrectTrajectory;
 public:
  StefanFish(SimulationData&s, ArgumentParser&p, Real C[2]);
  void create(const vector<BlockInfo>& vInfo) override;
};
