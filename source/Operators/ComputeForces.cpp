//
//  CubismUP_2D
//  Copyright (c) 2018 CSE-Lab, ETH Zurich, Switzerland.
//  Distributed under the terms of the MIT license.
//
//  Created by Guido Novati (novatig@ethz.ch).
//

#include "ComputeForces.h"
#include "../Shape.h"

using UDEFMAT = Real[VectorBlock::sizeY][VectorBlock::sizeX][2];

// TODO VARIABLE DENSITY
//
void ComputeForces::operator()(const double dt)
{
  const Real NUoH = sim.nu / sim.getH(); // 2 nu / 2 h
  static constexpr int stenBeg[3] = {-1,-1, 0}, stenEnd[3] = { 2, 2, 1};

  #pragma omp parallel
  {
    VectorLab  velLab;  velLab.prepare(*(sim.vel ), stenBeg, stenEnd, 0);
    ScalarLab presLab; presLab.prepare(*(sim.pres), stenBeg, stenEnd, 0);

    for(const Shape * const shape : sim.shapes)
    {
      const std::vector<ObstacleBlock*> & OBLOCK = shape->obstacleBlocks;
      const Real Cx = shape->centerOfMass[0], Cy = shape->centerOfMass[1];
      const Real vel_norm = std::sqrt(shape->u*shape->u + shape->v*shape->v);
      const Real vel_unit[2] = {
        vel_norm>0? (Real) shape->u / vel_norm : (Real)0,
        vel_norm>0? (Real) shape->v / vel_norm : (Real)0
      };

      #pragma omp for schedule(static)
      for (size_t i=0; i < Nblocks; ++i)
      {
        ObstacleBlock * const O = OBLOCK[velInfo[i].blockID];
        if (O == nullptr) continue;
        assert(O->filled);

         velLab.load( velInfo[i],0); const auto & __restrict__ V =  velLab;
        presLab.load(presInfo[i],0); const auto & __restrict__ P = presLab;

        for(size_t k = 0; k < O->n_surfPoints; ++k)
        {
          const int ix = O->surface[k]->ix, iy = O->surface[k]->iy;
          const std::array<Real,2> p = velInfo[i].pos<Real>(ix, iy);

          //shear stresses
          const Real D11 = NUoH   * (V(ix+1,iy  ).u[0] - V(ix-1,iy  ).u[0]);
          const Real D22 = NUoH   * (V(ix  ,iy+1).u[1] - V(ix  ,iy-1).u[1]);
          const Real D12 = NUoH/2 * (V(ix  ,iy+1).u[0] - V(ix  ,iy-1).u[0]
                                    +V(ix+1,iy  ).u[1] - V(ix-1,iy  ).u[1]);

          //normals computed with Towers 2009
          // Actually using the volume integral, since (/iint -P /hat{n} dS) =
          // (/iiint - /nabla P dV). Also, P*/nabla /Chi = /nabla P
          // penalty-accel and surf-force match up if resolution is high enough
          const Real normX = O->surface[k]->dchidx; // *h^2 (alreadt pre-
          const Real normY = O->surface[k]->dchidy; // -multiplied in dchidx/y)
          const Real fXV = D11*normX + D12*normY, fXP = - P(ix,iy).s * normX;
          const Real fYV = D12*normX + D22*normY, fYP = - P(ix,iy).s * normY;
          const Real fXT = fXV + fXP, fYT = fYV + fYP;
          //store:
          O-> P[k] = P(ix,iy).s;
          O->pX[k] = p[0];
          O->pY[k] = p[1];
          O->fX[k] = fXT;
          O->fY[k] = fYT;
          O->vx[k] = V(ix,iy).u[0];
          O->vy[k] = V(ix,iy).u[1];
          O->vxDef[k] = O->udef[iy][ix][0];
          O->vyDef[k] = O->udef[iy][ix][1];
          //perimeter:
          O->perimeter += std::sqrt(normX*normX + normY*normY);
          O->circulation += normX * O->vy[k] - normY * O->vx[k];
          //forces (total, visc, pressure):
          O->forcex += fXT;
          O->forcey += fYT;
          O->forcex_V += fXV;
          O->forcey_V += fYV;
          O->forcex_P += fXP;
          O->forcey_P += fYP;
          //torque:
          O->torque   += (p[0] - Cx) * fYT - (p[1] - Cy) * fXT;
          O->torque_P += (p[0] - Cx) * fYP - (p[1] - Cy) * fXP;
          O->torque_V += (p[0] - Cx) * fYV - (p[1] - Cy) * fXV;
          //thrust, drag:
          const Real forcePar = fXT * vel_unit[0] + fYT * vel_unit[1];
          O->thrust += .5*(forcePar + std::fabs(forcePar));
          O->drag   -= .5*(forcePar - std::fabs(forcePar));

          //power output (and negative definite variant which ensures no elastic energy absorption)
          // This is total power, for overcoming not only deformation, but also the oncoming velocity. Work done by fluid, not by the object (for that, just take -ve)
          const Real powOut = fXT * O->vx[k]    + fYT * O->vy[k];
          //deformation power output (and negative definite variant which ensures no elastic energy absorption)
          const Real powDef = fXT * O->vxDef[k] + fYT * O->vyDef[k];
          O->Pout        += powOut;
          O->defPower    += powDef;
          O->PoutBnd     += std::min((Real)0, powOut);
          O->defPowerBnd += std::min((Real)0, powDef);
        }
      }
    }
  }

  // finalize partial sums
  for(Shape * const shape : sim.shapes) shape->computeForces();
}

ComputeForces::ComputeForces(SimulationData& s) : Operator(s) { }
