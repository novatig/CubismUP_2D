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


#include "ShapeLibrary.h"
#include "ShapesSimple.h"

void Disk::create(const vector<BlockInfo>& vInfo)
{
  const Real h =  vInfo[0].h_gridpoint;
  for(auto & entry : obstacleBlocks) delete entry;
  obstacleBlocks.clear();
  obstacleBlocks = std::vector<ObstacleBlock*> (vInfo.size(), nullptr);

  #pragma omp parallel
  {
    FillBlocks_Cylinder kernel(radius, h, center, rhoS);

    #pragma omp for schedule(dynamic)
    for(size_t i=0; i<vInfo.size(); i++)
      if(kernel._is_touching(vInfo[i]))
      {
        assert(obstacleBlocks[vInfo[i].blockID] == nullptr);
        obstacleBlocks[vInfo[i].blockID] = new ObstacleBlock;
        obstacleBlocks[vInfo[i].blockID]->clear(); //memset 0

        const auto pos = obstacleBlocks[vInfo[i].blockID];
        assert(pos not_eq nullptr);

        FluidBlock& b = *(FluidBlock*)vInfo[i].ptrBlock;
        kernel(vInfo[i], b, pos);
      }
  }

  removeMoments(vInfo);
  for (auto & o : obstacleBlocks) if(o not_eq nullptr) o->allocate_surface();
}

void Disk::computeVelocities()
{
  Shape::computeVelocities();
  if(tAccel > 0) {
    if(bForcedx && sim.time < tAccel) u = (sim.time/tAccel)*forcedu;
    if(bForcedy && sim.time < tAccel) v = (sim.time/tAccel)*forcedv;
  }
}
void HalfDisk::computeVelocities()
{
  Shape::computeVelocities();
  if(tAccel > 0) {
    if(bForcedx && sim.time < tAccel) u = (sim.time/tAccel)*forcedu;
    if(bForcedy && sim.time < tAccel) v = (sim.time/tAccel)*forcedv;
  }
}

void HalfDisk::create(const vector<BlockInfo>& vInfo)
{
  const Real h =  vInfo[0].h_gridpoint;
  for(auto & entry : obstacleBlocks) delete entry;
  obstacleBlocks.clear();
  obstacleBlocks = std::vector<ObstacleBlock*> (vInfo.size(), nullptr);

  #pragma omp parallel
  {
    FillBlocks_HalfCylinder kernel(radius, h, center, rhoS, orientation);

    #pragma omp for schedule(dynamic)
    for(size_t i=0; i<vInfo.size(); i++)
      if(kernel._is_touching(vInfo[i]))
      {
        assert(obstacleBlocks[vInfo[i].blockID] == nullptr);
        obstacleBlocks[vInfo[i].blockID] = new ObstacleBlock;
        obstacleBlocks[vInfo[i].blockID]->clear(); //memset 0

        const auto pos = obstacleBlocks[vInfo[i].blockID];
        assert(pos not_eq nullptr);

        FluidBlock& b = *(FluidBlock*)vInfo[i].ptrBlock;
        kernel(vInfo[i], b, pos);
      }
  }

  removeMoments(vInfo);
  for (auto & o : obstacleBlocks) if(o not_eq nullptr) o->allocate_surface();
}

void Ellipse::create(const vector<BlockInfo>& vInfo)
{
  const Real h =  vInfo[0].h_gridpoint;
  for(auto & entry : obstacleBlocks) delete entry;
  obstacleBlocks.clear();
  obstacleBlocks = std::vector<ObstacleBlock*> (vInfo.size(), nullptr);

  #pragma omp parallel
  {
    FillBlocks_Ellipse kernel(semiAxis[0], semiAxis[1], h, center, orientation, rhoS);

    #pragma omp for schedule(dynamic)
    for(size_t i=0; i<vInfo.size(); i++)
      if(kernel._is_touching(vInfo[i]))
      {
        assert(obstacleBlocks[vInfo[i].blockID] == nullptr);
        obstacleBlocks[vInfo[i].blockID] = new ObstacleBlock;
        obstacleBlocks[vInfo[i].blockID]->clear(); //memset 0

        const auto pos = obstacleBlocks[vInfo[i].blockID];
        assert(pos not_eq nullptr);

        FluidBlock& b = *(FluidBlock*)vInfo[i].ptrBlock;
        kernel(vInfo[i], b, pos);
      }
  }

  const FillBlocks_EllipseFinalize finalize(h, rhoS);
  compute(finalize, vInfo);

  for (auto & o : obstacleBlocks) if(o not_eq nullptr) o->allocate_surface();
}

void DiskVarDensity::create(const vector<BlockInfo>& vInfo)
{
  const Real h =  vInfo[0].h_gridpoint;
  for(auto & entry : obstacleBlocks) delete entry;
  obstacleBlocks.clear();
  obstacleBlocks = std::vector<ObstacleBlock*> (vInfo.size(), nullptr);

  #pragma omp parallel
  {
    FillBlocks_VarRhoCylinder kernel(radius, h, center, rhoTop, rhoBot, orientation);

    #pragma omp for schedule(dynamic)
    for(size_t i=0; i<vInfo.size(); i++)
      if(kernel._is_touching(vInfo[i]))
      {
        assert(obstacleBlocks[vInfo[i].blockID] == nullptr);
        obstacleBlocks[vInfo[i].blockID] = new ObstacleBlock;
        obstacleBlocks[vInfo[i].blockID]->clear(); //memset 0

        const auto pos = obstacleBlocks[vInfo[i].blockID];
        assert(pos not_eq nullptr);

        FluidBlock& b = *(FluidBlock*)vInfo[i].ptrBlock;
        kernel(vInfo[i], b, pos);
      }
  }

  removeMoments(vInfo);
  for (auto & o : obstacleBlocks) if(o not_eq nullptr) o->allocate_surface();
}

void EllipseVarDensity::create(const vector<BlockInfo>& vInfo)
{
  const Real h =  vInfo[0].h_gridpoint;
  for(auto & entry : obstacleBlocks) delete entry;
  obstacleBlocks.clear();
  obstacleBlocks = std::vector<ObstacleBlock*> (vInfo.size(), nullptr);

  #pragma omp parallel
  {
    FillBlocks_Ellipse kernel(semiAxisX, semiAxisY, h, center, orientation, rhoTop);

    #pragma omp for schedule(dynamic)
    for(size_t i=0; i<vInfo.size(); i++)
      if(kernel._is_touching(vInfo[i]))
      {
        assert(obstacleBlocks[vInfo[i].blockID] == nullptr);
        obstacleBlocks[vInfo[i].blockID] = new ObstacleBlock;
        obstacleBlocks[vInfo[i].blockID]->clear(); //memset 0

        const auto pos = obstacleBlocks[vInfo[i].blockID];
        assert(pos not_eq nullptr);

        FluidBlock& b = *(FluidBlock*)vInfo[i].ptrBlock;
        kernel(vInfo[i], b, pos);
      }
  }

  const FillBlocks_VarRhoEllipseFinalize finalize(h, center, rhoTop, rhoBot, orientation);
  compute(finalize, vInfo);
  removeMoments(vInfo);
  for (auto & o : obstacleBlocks) if(o not_eq nullptr) o->allocate_surface();
}
