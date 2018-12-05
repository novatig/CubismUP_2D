//
//  ProcessOperatorsOMP.h
//  CubismUP_2D
//
//  Created by Christian Conti on 1/8/15.
//  Copyright (c) 2015 ETHZ. All rights reserved.
//

#pragma once

#include "../Operator.h"

class findMaxU
{
  SimulationData& sim;
  const vector<BlockInfo>& velInfo = sim.vel->getBlocksInfo();
  const size_t Nblocks = velInfo.size();
 public:
  findMaxU(SimulationData& s) : sim(s) { }

  Real run() const;

  string getName()
  {
    return "findMaxU";
  }
};

class Checker
{
  SimulationData& sim;
  const vector<BlockInfo>& velInfo = sim.vel->getBlocksInfo();
  const size_t Nblocks = velInfo.size();
 public:
  Checker(SimulationData& s) : sim(s) { }

  void run(std::string when) const;

  string getName()
  {
    return "Checker";
  }
};

class IC : public Operator
{
  public:
  IC(SimulationData& s) : Operator(s) { }

  void operator()(const double dt);

  string getName()
  {
    return "IC";
  }
};

class ApplyObjVel : public Operator
{
  public:
  ApplyObjVel(SimulationData& s) : Operator(s) { }

  void operator()(const double dt);

  string getName()
  {
    return "ApplyObjVel";
  }
};
