//
//  CoordinatorDiffusion.h
//  CubismUP_2D
//
//  Created by Christian Conti on 3/27/15.
//  Copyright (c) 2015 ETHZ. All rights reserved.
//

/*
This operator assumes that obects have put signed distance on the grid
*/
#include "PutObjectsOnGrid.h"
#include "../Shape.h"

static constexpr Real EPS = std::numeric_limits<Real>::epsilon();

void PutObjectsOnGrid::putChiOnGrid(Shape * const shape) const
{
  const std::vector<ObstacleBlock*>& OBLOCK = shape->obstacleBlocks;

  const Real h = sim.getH(), i2h = 0.5/h, fac = 0.5*h;
  #pragma omp parallel
  {
    static constexpr int stenBeg[3] = {-1,-1, 0}, stenEnd[3] = { 2, 2, 1};
    ScalarLab distlab; distlab.prepare(*(sim.tmp), stenBeg, stenEnd, 0);

    #pragma omp for schedule(static)
    for (size_t i=0; i < Nblocks; i++)
    {
      if(OBLOCK[chiInfo[i].blockID] == nullptr) continue; //obst not in block

      distlab.load(tmpInfo[i], 0); // loads signed distance field with ghosts
      const ScalarLab& __restrict__ SDIST = distlab;
      auto & __restrict__ CHI  = *(ScalarBlock*) chiInfo[i].ptrBlock; // dest

      for(int iy=0; iy<VectorBlock::sizeY; iy++)
      for(int ix=0; ix<VectorBlock::sizeX; ix++)
      {
        if (SDIST(ix,iy).s > +2*h || SDIST(ix,iy).s < -2*h)
        {
          const Real H = SDIST(ix, iy).s>0 ? 1 : 0;
          CHI(ix,iy).s = std::max(H, CHI(ix,iy).s);
          OBLOCK[chiInfo[i].blockID]->write(ix, iy, 1, H, 0, 0, 0);
          continue;
        }

        const Real distPx = SDIST(ix+1,iy  ).s, distMx = SDIST(ix-1,iy  ).s;
        const Real distPy = SDIST(ix  ,iy+1).s, distMy = SDIST(ix  ,iy-1).s;
        const Real IplusX = distPx<0? 0:distPx, IminuX = distMx<0? 0:distMx;
        const Real IplusY = distPy<0? 0:distPy, IminuY = distMy<0? 0:distMy;

        #if 0
         const Real HplusX = std::fabs(distPx)<EPS? (Real).5 : (distPx<0?0:1);
         const Real HminuX = std::fabs(distMx)<EPS? (Real).5 : (distMx<0?0:1);
         const Real HplusY = std::fabs(distPy)<EPS? (Real).5 : (distPy<0?0:1);
         const Real HminuY = std::fabs(distMy)<EPS? (Real).5 : (distMy<0?0:1);
        #else
         const Real HplusX = distPx < 0 ? 0 : 1, HminuX = distMx < 0 ? 0 : 1;
         const Real HplusY = distPy < 0 ? 0 : 1, HminuY = distMy < 0 ? 0 : 1;
        #endif

        const Real gradIX = i2h*(IplusX-IminuX), gradIY = i2h*(IplusY-IminuY);
        const Real gradUX = i2h*(distPx-distMx), gradUY = i2h*(distPy-distMy);
        const Real gradHX =     (HplusX-HminuX), gradHY =     (HplusY-HminuY);

        const Real gradUSq = gradUX * gradUX + gradUY * gradUY;
        const Real denum = 1 / ( gradUSq<EPS ? EPS : gradUSq );
        const Real H =     (gradIX*gradUX + gradIY*gradUY) * denum;
        const Real D = fac*(gradHX*gradUX + gradHY*gradUY) * denum;
        OBLOCK[chiInfo[i].blockID]->write(ix, iy, 1, H, D, gradUX, gradUY);
        CHI(ix,iy).s = std::max(H, CHI(ix,iy).s);
      }
    }
  }

  for (auto & o : OBLOCK) if(o not_eq nullptr) o->allocate_surface();
}

void PutObjectsOnGrid::putObjectVelOnGrid(Shape * const shape) const
{
  const std::vector<ObstacleBlock*>& OBLOCK = shape->obstacleBlocks;

  const double u_s = shape->u, v_s = shape->v, omega_s = shape->omega;
  const double Cx = shape->centerOfMass[0], Cy = shape->centerOfMass[1];

  #pragma omp parallel for schedule(static)
  for (size_t i=0; i < Nblocks; i++)
  {
    if(OBLOCK[uDefInfo[i].blockID] == nullptr) continue; //obst not in block

    using UDEFMAT = Real[VectorBlock::sizeY][VectorBlock::sizeX][2];
    using CHI_MAT = Real[VectorBlock::sizeY][VectorBlock::sizeX];

    UDEFMAT & __restrict__ udef = OBLOCK[uDefInfo[i].blockID]->udef;
    CHI_MAT & __restrict__ chi  = OBLOCK[uDefInfo[i].blockID]->chi;
    auto & __restrict__ UDEF = *(VectorBlock*)uDefInfo[i].ptrBlock; // dest
    for(int iy=0; iy<VectorBlock::sizeY; iy++)
    for(int ix=0; ix<VectorBlock::sizeX; ix++)
      if ( chi[iy][ix] > 0 )
      { //plus equal in case of overlapping objects
        Real p[2]; uDefInfo[i].pos(p, ix, iy);
        UDEF(ix,iy).u[0] += u_s - omega_s*(p[1]-Cy) + udef[iy][ix][0];
        UDEF(ix,iy).u[1] += v_s + omega_s*(p[0]-Cx) + udef[iy][ix][1];
      }
  }
}

// sets pRHS to div(F_{t}) and tmpV to (F_{t})
void PutObjectsOnGrid::presRHS_step1(const double dt) const
{
  const std::vector<BlockInfo>& tmpVInfo = sim.tmpV->getBlocksInfo();
  const std::vector<BlockInfo>& pRHSInfo = sim.pRHS->getBlocksInfo();
  static constexpr int stenBeg[3] = {-1,-1, 0}, stenEnd[3] = {2,2,1};
  const Real i2h = 0.5/sim.getH(), rk23fac = 0.5 * dt;

  #pragma omp parallel
  {
    ScalarLab chi_lab; chi_lab.prepare(*(sim.chi  ), stenBeg, stenEnd, 0);
    VectorLab fpenlab; fpenlab.prepare(*(sim.force), stenBeg, stenEnd, 0);

    #pragma omp for schedule(static)
    for (size_t i=0; i < Nblocks; i++) {
      chi_lab.load(chiInfo[i], 0); // loads chi_t field with ghosts
      fpenlab.load(forceInfo[i], 0); // loads penl force field with ghosts
      const ScalarLab& __restrict__ X = chi_lab;
      const VectorLab& __restrict__ F = fpenlab;

      VectorBlock & __restrict__ TMPV = *(VectorBlock*)tmpVInfo[i].ptrBlock;
      ScalarBlock & __restrict__ PRHS = *(ScalarBlock*)pRHSInfo[i].ptrBlock;

      for(int iy=0; iy<VectorBlock::sizeY; iy++)
      for(int ix=0; ix<VectorBlock::sizeX; ix++) {

        const Real cpx = X(ix+1, iy).s, cpy = X(ix, iy+1).s;
        const Real clx = X(ix-1, iy).s, cly = X(ix, iy-1).s;
        const Real fpx = F(ix+1, iy).u[0], gpy = F(ix, iy+1).u[0];
        const Real flx = F(ix-1, iy).u[1], gly = F(ix, iy-1).u[1];

        TMPV(ix,iy).u[0] = rk23fac * F(ix,iy).u[0] * X(ix,iy).s;
        TMPV(ix,iy).u[1] = rk23fac * F(ix,iy).u[1] * X(ix,iy).s;
        PRHS(ix,iy).s = i2h*((cpx*fpx - clx*flx) + (cpy*gpy - cly*gly));
      }
    }
  }
}

// computes: - div(\chi_{t+1} Udef_{t+1})
void PutObjectsOnGrid::presRHS_step2(const double dt) const
{
  const std::vector<BlockInfo>& pRHSInfo = sim.pRHS->getBlocksInfo();
  static constexpr int stenBeg[3] = {-1,-1, 0}, stenEnd[3] = {2,2,1};
  const Real divFac = 1.0 / ( 2 * sim.getH() ) / dt;

  #pragma omp parallel
  {
    ScalarLab chi_lab; chi_lab.prepare(*(sim.chi ), stenBeg, stenEnd, 0);
    VectorLab udeflab; udeflab.prepare(*(sim.uDef), stenBeg, stenEnd, 0);

    #pragma omp for schedule(static)
    for (size_t i=0; i < Nblocks; i++) {
      chi_lab.load(chiInfo[i], 0); // loads signed distance field with ghosts
      udeflab.load(uDefInfo[i], 0); // loads def velocity field with ghosts
      const ScalarLab& __restrict__ CHI  = chi_lab;
      const VectorLab& __restrict__ UDEF = udeflab;

      ScalarBlock & __restrict__ RHS = *(ScalarBlock*)pRHSInfo[i].ptrBlock;

      for(int iy=0; iy<VectorBlock::sizeY; iy++)
      for(int ix=0; ix<VectorBlock::sizeX; ix++) {
        const Real cpx = CHI(ix+1, iy).s, cpy = CHI(ix, iy+1).s;
        const Real clx = CHI(ix-1, iy).s, cly = CHI(ix, iy-1).s;
        const Real upx = UDEF(ix+1, iy).u[0], vpy = UDEF(ix, iy+1).u[0];
        const Real ulx = UDEF(ix-1, iy).u[1], vly = UDEF(ix, iy-1).u[1];
        RHS(ix,iy).s -= divFac*( (cpx*upx - clx*ulx) + (cpy*vpy - cly*vly) );
      }
    }
  }
}

void PutObjectsOnGrid::operator()(const double dt)
{
  // 1) put div(f^t_penl) in pres RHS and f^t_penl in tmpV for vel step
  sim.startProfiler("PutObjectsOnGrid_presRHS_step1");
  presRHS_step1(dt);
  sim.stopProfiler();

  // 2) clear chi^t and udef^t
  sim.startProfiler("PutObjectsOnGrid_clear");
  std::cout << "nblocks = " << Nblocks << std::endl;
  #pragma omp parallel for schedule(static)
  for (size_t i=0; i < Nblocks; i++) {
    ScalarBlock & CHI  = *(ScalarBlock*) chiInfo[i].ptrBlock;
    VectorBlock & UDEF = *(VectorBlock*)uDefInfo[i].ptrBlock;
    ScalarBlock & TMP  = *(ScalarBlock*) tmpInfo[i].ptrBlock;
    UDEF.clear();
    CHI.clear();
    TMP.set(-1);
    for(int iy=0; iy<VectorBlock::sizeY; iy++)
    for(int ix=0; ix<VectorBlock::sizeX; ix++) assert(TMP(ix,iy).s<0);
  }
  sim.stopProfiler();

  // 3) put objects' signed dist function and udef on the obstacle blocks:
  sim.startProfiler("PutObjectsOnGrid_create");
  for(const auto& shape : sim.shapes) shape->create(tmpInfo);
  sim.stopProfiler();

  // 4) for each obstacle, from signed distance, put new chi on blocks
  sim.startProfiler("PutObjectsOnGrid_putChiOnGrid");
  for(const auto& shape : sim.shapes) putChiOnGrid( shape );
  sim.stopProfiler();

  // 5) remove moments from characteristic function and put on grid U_s
  sim.startProfiler("PutObjectsOnGrid_putObjectVelOnGrid");
  for(const auto& shape : sim.shapes) {
    shape->removeMoments(chiInfo); // now that we have CHI, remove moments
    putObjectVelOnGrid( shape ); // put actual vel on the object vel grid
  }
  sim.stopProfiler();

  // 6) compute second component of pressure RHS: - div(\chi^{t+1} Udef^{t+1})
  sim.startProfiler("PutObjectsOnGrid_presRHS_step2");
  presRHS_step2(dt);
  sim.stopProfiler();

  // 7) prepare force: multiply previous penl force by 0 if new chi is 0
  sim.startProfiler("PutObjectsOnGrid_guess");
  #pragma omp parallel for schedule(static)
  for (size_t i=0; i < Nblocks; i++) {
    const auto & CHI  = *(ScalarBlock*) chiInfo[i].ptrBlock; // dest
    auto & FPNL = *(VectorBlock*)forceInfo[i].ptrBlock;
    for(int iy=0; iy<VectorBlock::sizeY; ++iy)
    for(int ix=0; ix<VectorBlock::sizeX; ++ix) {
      FPNL(ix,iy).u[0] *= CHI(ix,iy).s > 0;
      FPNL(ix,iy).u[1] *= CHI(ix,iy).s > 0;
    }
  }
  sim.stopProfiler();
}
