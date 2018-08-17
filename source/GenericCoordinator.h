//
//  GenericCoordinator.h
//  CubismUP_2D
//
//  This class serves as the interface for a coordinator object
//  A coordinator object schedules the processing of blocks with its operator
//
//  Created by Christian Conti on 3/27/15.
//  Copyright (c) 2015 ETHZ. All rights reserved.
//

#pragma once

#include "Definitions.h"
#include "GenericOperator.h"

class GenericCoordinator
{
protected:
  SimulationData& sim;
  const vector<BlockInfo>& vInfo = sim.grid->getBlocksInfo();

  inline void check(string infoText)
  {
    #ifndef NDEBUG
    #pragma omp parallel for schedule(static)
    for(size_t i=0; i<vInfo.size(); i++)
    {
      BlockInfo info = vInfo[i];
      FluidBlock& b = *(FluidBlock*)info.ptrBlock;

      for(int iy=0; iy<FluidBlock::sizeY; ++iy)
      for(int ix=0; ix<FluidBlock::sizeX; ++ix)
      {
        if (std::isnan(b(ix,iy).invRho) || std::isnan(b(ix,iy).u) ||
        std::isnan(b(ix,iy).v) || std::isnan(b(ix,iy).p))
          cout << infoText.c_str() <<endl;
        //if (b(ix,iy).invRho <= 0) cout << infoText.c_str() << endl;

        //assert(b(ix,iy).invRho > 0);
        assert(!std::isnan(b(ix,iy).invRho)); assert(!std::isnan(b(ix,iy).u));
        assert(!std::isnan(b(ix,iy).v)); assert(!std::isnan(b(ix,iy).p));
        assert(!std::isnan(b(ix,iy).pOld)); assert(!std::isnan(b(ix,iy).tmpU));
        assert(!std::isnan(b(ix,iy).tmpV)); assert(!std::isnan(b(ix,iy).tmp));
        assert(b(ix,iy).invRho < 1e10); assert(b(ix,iy).u < 1e10);
        assert(b(ix,iy).v < 1e10); assert(b(ix,iy).p < 1e10);
      }
    }
    #endif
  }

public:
  GenericCoordinator(SimulationData& s) : sim(s) { }
  virtual ~GenericCoordinator() {}
  virtual void operator()(const double dt) = 0;

  virtual string getName() = 0;
};
