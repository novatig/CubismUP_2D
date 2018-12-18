//
//  CubismUP_2D
//  Copyright (c) 2018 CSE-Lab, ETH Zurich, Switzerland.
//  Distributed under the terms of the MIT license.
//
//  Created by Guido Novati (novatig@ethz.ch).
//


#pragma once
#include "ObstacleBlock.h"

struct OperatorComputeForces
{
  const int stencil_start[3] = {-1, -1, 0}, stencil_end[3] = {2, 2, 1};
  StencilInfo stencil;
  const Real NU, *vel_unit;
  const double *CM;

  OperatorComputeForces(const Real nu, const Real* vunit, const double* cm) :
   NU(nu), vel_unit(vunit), CM(cm)
  {
    stencil = StencilInfo(-1,-1,0, 2,2,1, false, 2, 0,1);
  }

  inline void operator()(Lab& l, const BlockInfo&info, FluidBlock&b, ObstacleBlock*const o) const
  {
    const Real NUoH = NU / info.h_gridpoint; // 2 nu / 2 h
    assert(o->filled);
    //loop over elements of block info that have nonzero gradChi
    for(size_t i=0; i<o->n_surfPoints; i++)
    {
      Real p[2];
      const int ix = o->surface[i]->ix, iy = o->surface[i]->iy;
      info.pos(p, ix, iy);

      //shear stresses
      const Real D11 =    NUoH*(l(ix+1,iy).u - l(ix-1,iy).u);
      const Real D22 =    NUoH*(l(ix,iy+1).v - l(ix,iy-1).v);
      const Real D12 = .5*NUoH*(l(ix,iy+1).u - l(ix,iy-1).u
                               +l(ix+1,iy).v - l(ix-1,iy).v);

      //normals computed with Towers 2009
      // Actually using the volume integral, since (\iint -P \hat{n} dS) = (\iiint -\nabla P dV). Also, P*\nabla\Chi = \nabla P
      // penalty-accel and surf-force match up if resolution is high enough (200 points per fish)
      const Real P = b(ix,iy).p;
      const Real normX = o->surface[i]->dchidx;
      const Real normY = o->surface[i]->dchidy; //*h^2 (premultiplied in dchidy)
      const Real fXV = D11 * normX + D12 * normY;
      const Real fYV = D12 * normX + D22 * normY;
      const Real fXP = -P * normX, fYP = -P * normY;
      const Real fXT = fXV+fXP, fYT = fYV+fYP;

      //store:
      o->P[i]=P; o->pX[i]=p[0]; o->pY[i]=p[1]; o->fX[i]=fXT; o->fY[i]=fYT;
      o->vxDef[i] = o->udef[iy][ix][0]; o->vx[i] = l(ix,iy).u;
      o->vyDef[i] = o->udef[iy][ix][1]; o->vy[i] = l(ix,iy).v;

      //perimeter:
      o->perimeter += sqrt(normX*normX + normY*normY);
      o->circulation += normX*o->vy[i] - normY*o->vx[i];
      //forces (total, visc, pressure):
      o->forcex   += fXT; o->forcey   += fYT;
      o->forcex_V += fXV; o->forcey_V += fYV;
      o->forcex_P += fXP; o->forcey_P += fYP;
      //torque:
      o->torque   += (p[0]-CM[0])*fYT - (p[1]-CM[1])*fXT;
      o->torque_P += (p[0]-CM[0])*fYP - (p[1]-CM[1])*fXP;
      o->torque_V += (p[0]-CM[0])*fYV - (p[1]-CM[1])*fXV;
      //thrust, drag:
      const Real forcePar = fXT*vel_unit[0] + fYT*vel_unit[1];
      o->thrust += .5*(forcePar + std::fabs(forcePar));
      o->drag   -= .5*(forcePar - std::fabs(forcePar));

      //power output (and negative definite variant which ensures no elastic energy absorption)
      // This is total power, for overcoming not only deformation, but also the oncoming velocity. Work done by fluid, not by the object (for that, just take -ve)
      const Real powOut = fXT*o->vx[i]    + fYT*o->vy[i];
      //deformation power output (and negative definite variant which ensures no elastic energy absorption)
      const Real powDef = fXT*o->vxDef[i] + fYT*o->vyDef[i];
      o->Pout        += powOut; o->PoutBnd     += std::min((Real)0., powOut);
      o->defPower    += powDef; o->defPowerBnd += std::min((Real)0., powDef);
    }
  }
};
