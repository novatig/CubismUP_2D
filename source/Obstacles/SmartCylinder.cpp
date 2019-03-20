//
//  CubismUP_2D
//  Copyright (c) 2018 CSE-Lab, ETH Zurich, Switzerland.
//  Distributed under the terms of the MIT license.
//
//  Created by Guido Novati (novatig@ethz.ch).
//


#include "SmartCylinder.h"
#include "ShapeLibrary.h"
#include "../Utils/BufferedLogger.h"

std::vector<double> SmartCylinder::state(const Real OX, const Real OY, const Real velScale) const
{
  const std::vector<BlockInfo>& velInfo = sim.vel->getBlocksInfo();
  // relative position 2
  // relative velocity (+ angular) 3
  // shear velocity sensors 8
  #ifdef SMART_ELLIPSE
    std::vector<double> ret (7 + 8, 0);
  #else
    std::vector<double> ret (5 + 8, 0);
  #endif
  int sind = 0;
  const Real timeScale = 2*radius / velScale, invh = 1 / velInfo[0].h_gridpoint;
  ret[sind++] = (centerOfMass[0] - OX)/(2*radius);
  ret[sind++] = (centerOfMass[1] - OY)/(2*radius);
  #ifdef SMART_ELLIPSE
    const Real cosA = std::cos(orientation), sinA = std::sin(orientation);
    ret[sind++] = cosA;
    ret[sind++] = sinA;
  #endif
  ret[sind++] = u / velScale;
  ret[sind++] = v / velScale;
  ret[sind++] = omega * timeScale;

  const auto holdsPoint = [&](const BlockInfo&info, const Real X, const Real Y)
  {
    Real MIN[2], MAX[2];
    info.pos(MIN, 0,0);
    info.pos(MAX, VectorBlock::sizeX-1, VectorBlock::sizeY-1);
    for(int i=0; i<2; ++i) {
        MIN[i] -= 0.5 * info.h_gridpoint; // pos returns cell centers
        MAX[i] += 0.5 * info.h_gridpoint; // we care about whole block
    }
    return X >= MIN[0] && Y >= MIN[1] && X <= MAX[0] && Y <= MAX[1];
  };

  #ifdef SMART_ELLIPSE
    for(int a = 0; a<4; a++)
  #else
    for(int a = 0; a<8; a++)
  #endif

  {
    #ifdef SMART_ELLIPSE
      const Real theta = a * 2 * M_PI / 4;
      const Real cosTheta = std::cos(theta), sinTheta = std::sin(theta);
      const Real sensrXX = -radius*cosTheta, sensrYY = SMART_AR*radius*sinTheta;
      const Real sensorX = centerOfMass[0] + cosA*sensrXX + sinA*sensrYY;
      const Real sensorY = centerOfMass[1] + cosA*sensrYY - sinA*sensrXX;
    #else
      const Real theta = a * 2 * M_PI / 8;
      const Real cosTheta = std::cos(theta), sinTheta = std::sin(theta);
      const Real sensorX = centerOfMass[0] - 1.1 * radius * cosTheta;
      const Real sensorY = centerOfMass[1] + 1.1 * radius * sinTheta;
    #endif

    for(size_t i=0; i<velInfo.size(); i++)
    {
      if(not holdsPoint(velInfo[i], sensorX, sensorY) ) continue;
      Real org[2]; velInfo[i].pos(org, 0, 0);
      const int indx = (int) std::round((sensorX - org[0])*invh);
      const int indy = (int) std::round((sensorY - org[1])*invh);
      const int clipIndX = std::min( std::max(0, indx), VectorBlock::sizeX-1);
      const int clipIndY = std::min( std::max(0, indy), VectorBlock::sizeY-1);
      Real p[2]; velInfo[i].pos(p, clipIndX, clipIndY);
      p[0] -= centerOfMass[0]; p[1] -= centerOfMass[1];
      VectorBlock& b = *(VectorBlock*)velInfo[i].ptrBlock;
      const double uDiff = b(clipIndX, clipIndY).u[0] - (u - omega*p[1]);
      const double vDiff = b(clipIndX, clipIndY).u[1] - (v + omega*p[0]);
      #ifdef SMART_ELLIPSE
        ret[sind + 2*a + 0] = uDiff; ret[sind + 2*a + 1] = vDiff;
      #else
        ret[sind + a] = (cosTheta*uDiff + sinTheta*vDiff) / velScale;
      #endif

      printf("sensor:[%f %f]->[%f %f] ind:[%d %d] val:%f\n",
      sensorX, sensorY, p[0], p[1], indx, indy, ret[sind + a]);
      break;
    }
  }

  return ret;
}

double SmartCylinder::reward(const Real velScale)
{
  const Real forceScale = std::pow(velScale, 2) * 2*radius;
  #if 0
    const Real enSpent = energy  / (forceScale * forceScale);
  #else
    //const Real timeScale = 2*radius / velScale;
    //const Real enSpent = energySurf  / (forceScale * velScale * timeScale);
    //printf("R: 20*%f + %f\n", energySurf, energy  / (forceScale * forceScale));
    const Real enSpent = energySurf + energy  / (forceScale * forceScale);
  #endif
  energy = 0;
  energySurf = 0;
  return enSpent;
}

void SmartCylinder::act(std::vector<double> action, const Real velScale)
{
  const Real forceScale = velScale*velScale * 2*radius;
  const Real torqueScale = forceScale * radius;
  assert(action.size() == 3);
  appliedForceX = action[0] * forceScale;
  appliedForceY = action[1] * forceScale;
  appliedTorque = action[2] * torqueScale;
}

void SmartCylinder::create(const std::vector<BlockInfo>& vInfo)
{
  const Real h =  vInfo[0].h_gridpoint;
  for(auto & entry : obstacleBlocks) delete entry;
  obstacleBlocks.clear();
  obstacleBlocks = std::vector<ObstacleBlock*> (vInfo.size(), nullptr);

  #pragma omp parallel
  {
    #ifdef SMART_ELLIPSE
    FillBlocks_Ellipse kernel(radius, SMART_AR*radius, h, center,orientation,1);
    #else
    FillBlocks_Cylinder kernel(radius, h, center, rhoS);
    #endif

    #pragma omp for schedule(dynamic, 1)
    for(size_t i=0; i<vInfo.size(); i++)
      if(kernel.is_touching(vInfo[i]))
      {
        assert(obstacleBlocks[vInfo[i].blockID] == nullptr);
        obstacleBlocks[vInfo[i].blockID] = new ObstacleBlock;
        //obstacleBlocks[vInfo[i].blockID]->clear(); //memset 0
        ScalarBlock& b = *(ScalarBlock*)vInfo[i].ptrBlock;
        kernel(vInfo[i], b, * obstacleBlocks[vInfo[i].blockID] );
      }
  }
}

void SmartCylinder::updateVelocity(double dt)
{
  if(not bForcedx) {
    const Real uNxt = fluidMomX / penalM;
    const Real FX = M * (uNxt - u) / dt;
    const Real accx = ( appliedForceX + FX ) / M;
    u = u + dt * accx;
  }

  if(not bForcedy) {
    const Real vNxt = fluidMomY / penalM;
    const Real FY = M * (vNxt - v) / dt;
    const Real accy = ( appliedForceY + FY ) / M;
    v = v + dt * accy;
  }

  if(not bBlockang) {
    const Real omegaNxt = fluidAngMom / penalJ;
    const Real TZ = J * (omegaNxt - omega) / dt;
    const Real acca = ( appliedTorque + TZ ) / J;
    omega = omega + dt * acca;
  }
  energy -= ( std::pow(appliedForceX, 2) + std::pow(appliedForceY, 2)
            + std::pow(appliedTorque/radius, 2) ) * dt;

  //if(bForcedx) { u = forcedu; cout << "bForcedx" << endl; }
  //if(bForcedy) v = forcedv;
  //if(bBlockang) omega = 0;
}

void SmartCylinder::updatePosition(double dt)
{
  #ifdef SMART_ELLIPSE // additional symmetry
    orientation = orientation> M_PI/2 ? orientation-M_PI : orientation;
    orientation = orientation<-M_PI/2 ? orientation+M_PI : orientation;
  #endif
  Shape::updatePosition(dt);
}

void SmartCylinder::computeForces()
{
  Shape::computeForces();
  //energySurf += PoutBnd * sim.dt;
  //energySurf += Pthrust/(Pthrust-std::min(Pout,(double)0)) * sim.dt;
  energySurf += Pthrust/(Pthrust-std::min(PoutBnd,(double)0)) * sim.dt;
}
