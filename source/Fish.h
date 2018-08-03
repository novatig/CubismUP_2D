//
//  IF2D_FishOperator.h
//  IF2D_ROCKS
//
//  Created by Wim van Rees on 17/11/14.
//
//

#pragma once

#include "Shape.h"
#include <vector>
#include <map>

class FishData;

class Fish: public Shape
{
 protected:
  const Real length, Tperiod, phaseShift;
  Real area_internal = 0, J_internal = 0;
  Real CoM_internal[2] ={0,0}, vCoM_internal[2] ={0,0};
  Real theta_internal = 0, angvel_internal = 0, angvel_internal_prev = 0;

  FishData * myFish = nullptr;
  //void apply_pid_corrections();

  Fish(SimulationData&s, ArgumentParser&p, Real C[2]) : Shape(s,p,C),
  length(p("-L").asDouble(0.1)), Tperiod(p("-T").asDouble(1)),
  phaseShift(p("-phi").asDouble(0))  {}
  virtual ~Fish();

 public:
  Real getCharLength() const override {
    return length;
  }
  void updatePosition(double dt) override;
  virtual void create(const vector<BlockInfo>& vInfo) override;
};
