//
//  CubismUP_2D
//  Copyright (c) 2018 CSE-Lab, ETH Zurich, Switzerland.
//  Distributed under the terms of the MIT license.
//
//  Created by Guido Novati (novatig@ethz.ch).
//

#pragma once

#include "Definitions.h"

using CHI_MAT = Real[_BS_][_BS_];
using UDEFMAT = Real[_BS_][_BS_][2];

struct surface_data
{
  const int ix, iy;
  const Real dchidx, dchidy, delta;

  surface_data(const int _ix, const int _iy, const Real _dchidx, const Real _dchidy, const Real _delta) : ix(_ix), iy(_iy), dchidx(_dchidx), dchidy(_dchidy), delta(_delta)
  {}
};

struct ObstacleBlock
{
  static const int sizeX = _BS_;
  static const int sizeY = _BS_;

  // bulk quantities:
  Real chi[sizeY][sizeX];
  Real dist[sizeY][sizeX];
  Real rho[sizeY][sizeX];
  Real udef[sizeY][sizeX][2];

  //surface quantities:
  size_t n_surfPoints=0;
  bool filled = false;
  std::vector<surface_data*> surface;
  Real *pX=nullptr, *pY=nullptr, *P=nullptr, *fX=nullptr, *fY=nullptr;
  Real *vx=nullptr, *vy=nullptr, *vxDef=nullptr, *vyDef=nullptr;

  //additive quantities:
  Real perimeter = 0, forcex = 0, forcey = 0, forcex_P = 0, forcey_P = 0;
  Real forcex_V = 0, forcey_V = 0, torque = 0, torque_P = 0, torque_V = 0;
  Real drag = 0, thrust = 0, Pout=0, PoutBnd=0, defPower=0, defPowerBnd = 0;
  Real circulation = 0;

  ObstacleBlock()
  {
    clear();
    //rough estimate of surface cutting the block diagonally
    //with 2 points needed on each side of surface
    surface.reserve(4*_BS_);
  }
  ~ObstacleBlock()
  {
    clear_surface();
  }

  void clear_surface()
  {
    filled = false;
    n_surfPoints = 0;
    perimeter = forcex = forcey = forcex_P = forcey_P = 0;
    forcex_V = forcey_V = torque = torque_P = torque_V = drag = thrust = 0;
    Pout = PoutBnd = defPower = defPowerBnd = circulation = 0;

    for (auto & trash : surface) {
      if(trash == nullptr) continue;
      delete trash;
      trash = nullptr;
    }
    surface.clear();
    if(pX    not_eq nullptr){delete[] pX;    pX    = nullptr; }
    if(pY    not_eq nullptr){delete[] pY;    pY    = nullptr; }
    if(P     not_eq nullptr){delete[] P;     P     = nullptr; }
    if(fX    not_eq nullptr){delete[] fX;    fX    = nullptr; }
    if(fY    not_eq nullptr){delete[] fY;    fY    = nullptr; }
    if(vx    not_eq nullptr){delete[] vx;    vx    = nullptr; }
    if(vy    not_eq nullptr){delete[] vy;    vy    = nullptr; }
    if(vxDef not_eq nullptr){delete[] vxDef; vxDef = nullptr; }
    if(vyDef not_eq nullptr){delete[] vyDef; vyDef = nullptr; }
  }

  void clear()
  {
    clear_surface();
    memset(chi, 0, sizeof(Real)*sizeX*sizeY);
    memset(dist, 0, sizeof(Real)*sizeX*sizeY);
    memset(rho, 0, sizeof(Real)*sizeX*sizeY);
    memset(udef, 0, sizeof(Real)*sizeX*sizeY*2);
  }

  void write(const int ix, const int iy, const Real delta, const Real gradUX, const Real gradUY)
  {
    assert(!filled);

    if ( delta > 0 ) {
      n_surfPoints++;
      // multiply by cell area h^2 and by 0.5/h due to finite diff of gradHX
      const Real dchidx = -delta*gradUX, dchidy = -delta*gradUY;
      surface.push_back( new surface_data(ix, iy, dchidx, dchidy, delta) );
    }
  }

  void allocate_surface()
  {
    filled = true;
    assert(surface.size() == n_surfPoints);
    pX    = new Real[n_surfPoints]; pY    = new Real[n_surfPoints];
    fX    = new Real[n_surfPoints]; fY    = new Real[n_surfPoints];
    vx    = new Real[n_surfPoints]; vy    = new Real[n_surfPoints];
    vxDef = new Real[n_surfPoints]; vyDef = new Real[n_surfPoints];
    P = new Real[n_surfPoints];
  }

  void print(std::ofstream& pFile)
  {
    assert(filled);
    for(size_t i=0; i<n_surfPoints; i++) {
      float buf[]={(float)pX[i], (float)pY[i], (float)P[i], (float)fX[i],
        (float)fY[i], (float)vx[i], (float)vy[i], (float)vxDef[i],
        (float)vyDef[i], (float)surface[i]->dchidx, (float)surface[i]->dchidy};
      pFile.write((char*)buf, sizeof(float)*11);
    }
  }

  void printCSV(std::ofstream& pFile)
  {
    assert(filled);
    for(size_t i=0; i<n_surfPoints; i++)
      pFile<<pX[i]<<", "<<pY[i]<<", "<<P[i]<<", "<<fX[i]<<", "<<fY[i]<<", "
           <<vx[i]<<", "<<vy[i]<<", "<<vxDef[i]<<", "<<vyDef[i]<<", "
           <<surface[i]->dchidx<<", "<<surface[i]->dchidy<<"\n";
  }
};
