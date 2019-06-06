//
//  CubismUP_2D
//  Copyright (c) 2018 CSE-Lab, ETH Zurich, Switzerland.
//  Distributed under the terms of the MIT license.
//
//  Created by Guido Novati (novatig@ethz.ch).
//


#include "PressureSingle.h"
#include "../Poisson/PoissonSolver.h"
#include "../Shape.h"

using namespace cubism;

static constexpr double EPS = std::numeric_limits<double>::epsilon();
using CHI_MAT = Real[VectorBlock::sizeY][VectorBlock::sizeX];
using UDEFMAT = Real[VectorBlock::sizeY][VectorBlock::sizeX][2];

#define EXPL_INTEGRATE_MOM

void PressureSingle::integrateMomenta(Shape * const shape) const
{
  const std::vector<ObstacleBlock*> & OBLOCK = shape->obstacleBlocks;
  const Real Cx = shape->centerOfMass[0], Cy = shape->centerOfMass[1];
  const double hsq = std::pow(velInfo[0].h_gridpoint, 2);
  double PM=0, PJ=0, PX=0, PY=0, UM=0, VM=0, AM=0; //linear momenta

  #pragma omp parallel for schedule(dynamic,1) reduction(+:PM,PJ,PX,PY,UM,VM,AM)
  for(size_t i=0; i<Nblocks; i++)
  {
    const VectorBlock& __restrict__ VEL = *(VectorBlock*)velInfo[i].ptrBlock;

    if(OBLOCK[velInfo[i].blockID] == nullptr) continue;
    const CHI_MAT & __restrict__ rho = OBLOCK[velInfo[i].blockID]->rho;
    const CHI_MAT & __restrict__ chi = OBLOCK[velInfo[i].blockID]->chi;
    const UDEFMAT & __restrict__ udef = OBLOCK[velInfo[i].blockID]->udef;
    #ifndef EXPL_INTEGRATE_MOM
      const Real lambdt = sim.lambda * sim.dt;
    #endif

    for(int iy=0; iy<VectorBlock::sizeY; ++iy)
    for(int ix=0; ix<VectorBlock::sizeX; ++ix)
    {
      if (chi[iy][ix] <= 0) continue;
      const Real udiff[2] = {
        VEL(ix,iy).u[0] - udef[iy][ix][0], VEL(ix,iy).u[1] - udef[iy][ix][1]
      };
      #ifdef EXPL_INTEGRATE_MOM
        const Real F = hsq * rho[iy][ix] * chi[iy][ix];
      #else
        const Real Xlamdt = chi[iy][ix] * lambdt;
        const Real F = hsq * rho[iy][ix] * Xlamdt / (1 + Xlamdt);
      #endif
      double p[2]; velInfo[i].pos(p, ix, iy); p[0] -= Cx; p[1] -= Cy;
      PM += F;
      PJ += F * (p[0]*p[0] + p[1]*p[1]);
      PX += F * p[0];  PY += F * p[1];
      UM += F * udiff[0]; VM += F * udiff[1];
      AM += F * (p[0]*udiff[1] - p[1]*udiff[0]);
    }
  }

  shape->fluidAngMom = AM; shape->fluidMomX = UM; shape->fluidMomY = VM;
  shape->penalDX=PX; shape->penalDY=PY; shape->penalM=PM; shape->penalJ=PJ;
}

void PressureSingle::penalize(const double dt) const
{
  #pragma omp parallel for schedule(dynamic, 1)
  for (size_t i=0; i < Nblocks; i++)
  for (Shape * const shape : sim.shapes)
  {
    const std::vector<ObstacleBlock*>& OBLOCK = shape->obstacleBlocks;
    const ObstacleBlock*const o = OBLOCK[velInfo[i].blockID];
    if (o == nullptr) continue;

    const Real u_s = shape->u, v_s = shape->v, omega_s = shape->omega;
    const Real Cx = shape->centerOfMass[0], Cy = shape->centerOfMass[1];

    const CHI_MAT & __restrict__ X = o->chi;
    const UDEFMAT & __restrict__ UDEF = o->udef;
    const ScalarBlock& __restrict__ CHI = *(ScalarBlock*)chiInfo[i].ptrBlock;
          VectorBlock& __restrict__   V = *(VectorBlock*)velInfo[i].ptrBlock;

    for(int iy=0; iy<VectorBlock::sizeY; ++iy)
    for(int ix=0; ix<VectorBlock::sizeX; ++ix)
    {
      // What if multiple obstacles share a block? Do not write udef onto
      // grid if CHI stored on the grid is greater than obst's CHI.
      if(CHI(ix,iy).s > X[iy][ix]) continue;
      if(X[iy][ix] <= 0) continue; // no need to do anything

      Real p[2]; velInfo[i].pos(p, ix, iy); p[0] -= Cx; p[1] -= Cy;
      #ifndef EXPL_INTEGRATE_MOM
        const Real alpha = 1/(1 + sim.lambda * dt * X[iy][ix]);
      #else
        const Real alpha = 1 - X[iy][ix];
      #endif

      const Real US = u_s - omega_s * p[1] + UDEF[iy][ix][0];
      const Real VS = v_s + omega_s * p[0] + UDEF[iy][ix][1];
      V(ix,iy).u[0] = alpha*V(ix,iy).u[0] + (1-alpha)*US;
      V(ix,iy).u[1] = alpha*V(ix,iy).u[1] + (1-alpha)*VS;
    }
  }
}

void PressureSingle::updatePressureRHS(const double dt) const
{
  const Real h = sim.getH(), facDiv = 0.5*h/dt;
  static constexpr int stenBeg[3] = {-1,-1, 0}, stenEnd[3] = { 2, 2, 1};
  #pragma omp parallel
  {
    VectorLab velLab;  velLab.prepare( *(sim.vel),  stenBeg, stenEnd, 0);
    #pragma omp for schedule(static)
    for (size_t i=0; i < Nblocks; i++)
    {
      velLab.load(velInfo[i], 0); const auto & __restrict__ V   = velLab;
      ScalarBlock& __restrict__ TMP = *(ScalarBlock*) tmpInfo[i].ptrBlock;
      for(int iy=0; iy<VectorBlock::sizeY; ++iy)
      for(int ix=0; ix<VectorBlock::sizeX; ++ix) {
        const Real divVx  = V(ix+1,iy).u[0] - V(ix-1,iy).u[0];
        const Real divVy  = V(ix,iy+1).u[1] - V(ix,iy-1).u[1];
        TMP(ix, iy).s = facDiv*(divVx+divVy);
      }
    }
  }

  const size_t nShapes = sim.shapes.size();
  Real * sumRHS, * absRHS;
  sumRHS = (Real*) calloc(nShapes, sizeof(Real));
  absRHS = (Real*) calloc(nShapes, sizeof(Real));

  #pragma omp parallel reduction(+: sumRHS[:nShapes], absRHS[:nShapes])
  {
    VectorLab velLab;   velLab.prepare(*(sim.vel ), stenBeg, stenEnd, 0);
    ScalarLab chiLab;   chiLab.prepare(*(sim.chi ), stenBeg, stenEnd, 0);
    #pragma omp for schedule(static)
    for (size_t i=0; i < Nblocks; i++)
    for (size_t j=0; j < nShapes; j++)
    {
      const Shape * const shape = sim.shapes[j];
      const std::vector<ObstacleBlock*> & OBLOCK = shape->obstacleBlocks;
      const ObstacleBlock * const o = OBLOCK[uDefInfo[i].blockID];
      if (o == nullptr) continue;

      const Real u_s = shape->u, v_s = shape->v, omega_s = shape->omega;
      const Real Cx = shape->centerOfMass[0], Cy = shape->centerOfMass[1];

      velLab.load(  velInfo[i], 0); const auto & __restrict__ V    = velLab;
      chiLab.load(  chiInfo[i], 0); const auto & __restrict__ CHI  = chiLab;
      const CHI_MAT & __restrict__ chi = o->chi;
      const UDEFMAT & __restrict__ udef = o->udef;
      auto & __restrict__ TMP = *(ScalarBlock*) tmpInfo[i].ptrBlock;

      for(int iy=0; iy<VectorBlock::sizeY; ++iy)
      for(int ix=0; ix<VectorBlock::sizeX; ++ix)
      {
        if (chi[iy][ix] <= 0 || CHI(ix,iy).s > chi[iy][ix]) continue;
        Real p[2]; velInfo[i].pos(p, ix, iy); p[0] -= Cx; p[1] -= Cy;
        const Real dU = u_s - omega_s * p[1] + udef[iy][ix][0] - V(ix,iy).u[0];
        const Real dV = v_s + omega_s * p[0] + udef[iy][ix][1] - V(ix,iy).u[1];

        const Real divVx  = V(ix+1,iy).u[0] - V(ix-1,iy).u[0];
        const Real divVy  = V(ix,iy+1).u[1] - V(ix,iy-1).u[1];
        const Real gradXx = CHI(ix+1,iy).s - CHI(ix-1,iy).s;
        const Real gradXy = CHI(ix,iy+1).s - CHI(ix,iy-1).s;
        const Real srcPerimeter = facDiv * ( gradXx*dU + gradXy*dV );
        const Real srcBulk = - facDiv * CHI(ix,iy).s * ( divVx + divVy );
        sumRHS[j] += srcPerimeter + srcBulk;
        absRHS[j] += std::fabs(srcPerimeter);
        TMP(ix, iy).s += srcPerimeter + srcBulk;
      }
    }
  }

  for (size_t j=0; j < nShapes; j++)
  printf("sum of udef src terms %e\n", sumRHS[j]/std::max(absRHS[j], EPS));

  #pragma omp parallel
  {
    ScalarLab chiLab;   chiLab.prepare(*(sim.chi ), stenBeg, stenEnd, 0);

    #pragma omp for schedule(static)
    for (size_t i=0; i < Nblocks; i++)
    for (size_t j=0; j < nShapes; j++)
    {
      const Shape * const shape = sim.shapes[j];
      const Real corr = sumRHS[j] / std::max(absRHS[j], EPS);
      const std::vector<ObstacleBlock*>& OBLOCK = shape->obstacleBlocks;
      const ObstacleBlock*const o = OBLOCK[uDefInfo[i].blockID];
      if (o == nullptr) continue;

      const Real u_s = shape->u, v_s = shape->v, omega_s = shape->omega;
      const Real Cx = shape->centerOfMass[0], Cy = shape->centerOfMass[1];

      const CHI_MAT & __restrict__ chi = o->chi;
      const UDEFMAT & __restrict__ udef = o->udef;
      ScalarBlock& __restrict__ TMP = *(ScalarBlock*) tmpInfo[i].ptrBlock;
      chiLab.load(  chiInfo[i], 0); const auto & __restrict__ CHI  = chiLab;
      const VectorBlock& __restrict__ V = *(VectorBlock*) velInfo[i].ptrBlock;

      for(int iy=0; iy<VectorBlock::sizeY; ++iy)
      for(int ix=0; ix<VectorBlock::sizeX; ++ix)
      {
        if (chi[iy][ix] <= 0 || CHI(ix,iy).s > chi[iy][ix]) continue;
        Real p[2]; velInfo[i].pos(p, ix, iy); p[0] -= Cx; p[1] -= Cy;
        const Real dU = u_s - omega_s * p[1] + udef[iy][ix][0] - V(ix,iy).u[0];
        const Real dV = v_s + omega_s * p[0] + udef[iy][ix][1] - V(ix,iy).u[1];
        const Real gradXx = CHI(ix+1,iy).s - CHI(ix-1,iy).s;
        const Real gradXy = CHI(ix,iy+1).s - CHI(ix,iy-1).s;
        const Real srcPerimeter = facDiv * ( gradXx*dU + gradXy*dV );
        TMP(ix, iy).s -= corr * std::fabs(srcPerimeter);
      }
    }
  }

  free (sumRHS); free (absRHS);
}

void PressureSingle::pressureCorrection(const double dt) const
{
  const Real h = sim.getH(), pFac = -0.5*dt/h;//, invDt = 1/dt;//sim.lambda;

  #pragma omp parallel
  {
    static constexpr int stenBeg[3] = {-1,-1, 0}, stenEnd[3] = { 2, 2, 1};
    ScalarLab plab; plab.prepare(*(sim.pres), stenBeg, stenEnd, 0);

    #pragma omp for schedule(static)
    for (size_t i=0; i < Nblocks; i++)
    {
      plab.load(presInfo[i], 0); // loads pres field with ghosts
      const ScalarLab  &__restrict__   P = plab; // only this needs ghosts
            VectorBlock&__restrict__   V = *(VectorBlock*)  velInfo[i].ptrBlock;

      for(int iy=0; iy<VectorBlock::sizeY; ++iy)
      for(int ix=0; ix<VectorBlock::sizeX; ++ix)
      {
        // update vel field after most recent force and pressure response:
        V(ix,iy).u[0] = V(ix,iy).u[0] + pFac *(P(ix+1,iy).s-P(ix-1,iy).s);
        V(ix,iy).u[1] = V(ix,iy).u[1] + pFac *(P(ix,iy+1).s-P(ix,iy-1).s);
      }
    }
  }
}

void PressureSingle::operator()(const double dt)
{
  sim.startProfiler("integrateMomenta");
  for(Shape * const shape : sim.shapes) {
    integrateMomenta(shape);
    shape->updateVelocity(dt);
  }
  sim.stopProfiler();

  sim.startProfiler("Prhs");
  updatePressureRHS(dt);
  //fadeoutBorder(dt);
  sim.stopProfiler();

  pressureSolver->solve(tmpInfo, presInfo);

  sim.startProfiler("penalize");
  penalize(dt);
  sim.stopProfiler();

  sim.startProfiler("PCorrect");
  pressureCorrection(dt);
  sim.stopProfiler();
}

PressureSingle::PressureSingle(SimulationData& s) : Operator(s),
pressureSolver( PoissonSolver::makeSolver(s) ) { }

PressureSingle::~PressureSingle() {
  delete pressureSolver;
}
