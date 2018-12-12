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
  const Real h = sim.getH();
  const double u_s = shape->u, v_s = shape->v, omega_s = shape->omega;
  const double Cx = shape->centerOfMass[0], Cy = shape->centerOfMass[1];

  #pragma omp parallel for schedule(dynamic)
  for (size_t i=0; i < Nblocks; i++)
  {
    if(OBLOCK[uDefInfo[i].blockID] == nullptr) continue; //obst not in block

    using UDEFMAT = Real[VectorBlock::sizeY][VectorBlock::sizeX][2];
    //using CHI_MAT = Real[VectorBlock::sizeY][VectorBlock::sizeX];

    UDEFMAT & __restrict__ udef = OBLOCK[uDefInfo[i].blockID]->udef;
    //CHI_MAT & __restrict__ chi  = OBLOCK[uDefInfo[i].blockID]->chi;
    auto & __restrict__ UDEF = *(VectorBlock*)uDefInfo[i].ptrBlock; // dest
    const ScalarBlock&__restrict__ TMP  = *(ScalarBlock*) tmpInfo[i].ptrBlock;

    for(int iy=0; iy<VectorBlock::sizeY; iy++)
    for(int ix=0; ix<VectorBlock::sizeX; ix++)
      if (TMP(ix,iy).s > -3*h) //( chi[iy][ix] > 0 )
      { //plus equal in case of overlapping objects
        Real p[2]; uDefInfo[i].pos(p, ix, iy);
        UDEF(ix,iy).u[0] += u_s - omega_s*(p[1]-Cy) + udef[iy][ix][0];
        UDEF(ix,iy).u[1] += v_s + omega_s*(p[0]-Cx) + udef[iy][ix][1];
      }
  }
}

void PutObjectsOnGrid::operator()(const double dt)
{
  // TODO I NEED SIGNED DISTANCE PER OBSTACLE
  // 1) clear chi^t and udef^t
  sim.startProfiler("ObjGrid_clear");
  std::cout << "nblocks = " << Nblocks << std::endl;
  #pragma omp parallel for schedule(static)
  for (size_t i=0; i < Nblocks; i++) {
    ScalarBlock & CHI  = *(ScalarBlock*) chiInfo[i].ptrBlock;
    VectorBlock & UDEF = *(VectorBlock*)uDefInfo[i].ptrBlock;
    ScalarBlock & TMP  = *(ScalarBlock*) tmpInfo[i].ptrBlock;
    UDEF.clear();
    CHI.clear();
    TMP.set(-1);
  }
  sim.stopProfiler();

  // 2) put objects' signed dist function and udef on the obstacle blocks:
  sim.startProfiler("ObjGrid_make");
  for(const auto& shape : sim.shapes) shape->create(tmpInfo);
  sim.stopProfiler();

  // 3) for each obstacle, from signed distance, put new chi on blocks
  sim.startProfiler("ObjGrid_chi");
  for(const auto& shape : sim.shapes) putChiOnGrid( shape );
  sim.stopProfiler();

  // 4) remove moments from characteristic function and put on grid U_s
  sim.startProfiler("ObjGrid_uobj");
  for(const auto& shape : sim.shapes) {
    shape->removeMoments(chiInfo); // now that we have CHI, remove moments
    putObjectVelOnGrid( shape ); // put actual vel on the object vel grid
  }
  sim.stopProfiler();
}
