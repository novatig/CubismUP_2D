//
//  IF2D_StefanFishOperator.h
//  IF2D_ROCKS
//
//  Created by Wim van Rees on 25/11/14.
//
//

#pragma once
#include "Fish.h"
class CarlingFish: public Fish
{
 public:
  double getPhase(const double t) const;

  CarlingFish(SimulationData&s, ArgumentParser&p, double C[2]);
  void resetAll() override;
  void create(const vector<BlockInfo>& vInfo) override;
};
