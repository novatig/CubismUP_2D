//
//  CubismUP_2D
//  Copyright (c) 2018 CSE-Lab, ETH Zurich, Switzerland.
//  Distributed under the terms of the MIT license.
//
//  Created by Guido Novati (novatig@ethz.ch).
//


#include "PutObjectsOnGrid.h"
#include "../Shape.h"

using namespace cubism;

static constexpr double EPS = std::numeric_limits<double>::epsilon();

void PutObjectsOnGrid::putChiOnGrid(Shape * const shape) const
{
  const std::vector<ObstacleBlock*>& OBLOCK = shape->obstacleBlocks;
  double _x=0, _y=0, _m=0;
  //double udefoutflow=0, udefoutnorm=0; // , udefoutflow, udefoutnorm
  const Real h = sim.getH(), i2h = 0.5/h, fac = 0.5*h; // fac explained down
  #pragma omp parallel reduction(+ : _x, _y, _m)
  {
    static constexpr int stenBeg[3] = {-1,-1, 0}, stenEnd[3] = { 2, 2, 1};
    ScalarLab distlab; distlab.prepare(*(sim.tmp), stenBeg, stenEnd, 0);

    #pragma omp for schedule(dynamic, 1)
    for (size_t i=0; i < Nblocks; i++)
    {
      if(OBLOCK[chiInfo[i].blockID] == nullptr) continue; //obst not in block
      ObstacleBlock& o = * OBLOCK[chiInfo[i].blockID];

      distlab.load(tmpInfo[i], 0); // loads signed distance field with ghosts
      const ScalarLab& __restrict__ SDIST = distlab;
      auto & __restrict__ CHI  = *(ScalarBlock*)    chiInfo[i].ptrBlock;
      auto & __restrict__ IRHO = *(ScalarBlock*) invRhoInfo[i].ptrBlock;
      CHI_MAT & __restrict__ X = o.chi;
      const CHI_MAT & __restrict__ rho = o.rho;
      const CHI_MAT & __restrict__ sdf = o.dist;
      //UDEFMAT & __restrict__ udef = o.udef;

      for(int iy=0; iy<VectorBlock::sizeY; iy++)
      for(int ix=0; ix<VectorBlock::sizeX; ix++)
      {
        // here I read SDF to deal with obstacles sharing block
        if (sdf[iy][ix] > +2*h || sdf[iy][ix] < -2*h)
        {
          X[iy][ix] = sdf[iy][ix] > 0 ? 1 : 0;
        }
        else
        {
          const double distPx = ix+1 == _BS_ ? SDIST(ix+1,iy).s : sdf[iy][ix+1];
          const double distMx = ix   == 0    ? SDIST(ix-1,iy).s : sdf[iy][ix-1];
          const double distPy = iy+1 == _BS_ ? SDIST(ix,iy+1).s : sdf[iy+1][ix];
          const double distMy = iy   == 0    ? SDIST(ix,iy-1).s : sdf[iy-1][ix];
          const double IplusX = distPx<0? 0:distPx, IminuX = distMx<0? 0:distMx;
          const double IplusY = distPy<0? 0:distPy, IminuY = distMy<0? 0:distMy;

          const double gradIX= i2h*(IplusX-IminuX), gradIY= i2h*(IplusY-IminuY);
          const double gradUX= i2h*(distPx-distMx), gradUY= i2h*(distPy-distMy);

          const double gradUSq = gradUX * gradUX + gradUY * gradUY;
          const double denum = 1 / ( gradUSq<EPS ? EPS : gradUSq );
          X[iy][ix] = (gradIX*gradUX + gradIY*gradUY) * denum;
        }

        // an other partial
        if(X[iy][ix] >= CHI(ix,iy).s)
        {
           CHI(ix,iy).s = X[iy][ix];
          IRHO(ix,iy).s = X[iy][ix]/rho[iy][ix] + (1-X[iy][ix])*IRHO(ix,iy).s;
        }
        if(X[iy][ix] > 0)
        {
          double p[2]; chiInfo[i].pos(p, ix, iy);
          _x += rho[iy][ix] * X[iy][ix] * h*h * (p[0] - shape->centerOfMass[0]);
          _y += rho[iy][ix] * X[iy][ix] * h*h * (p[1] - shape->centerOfMass[1]);
          _m += rho[iy][ix] * X[iy][ix] * h*h;
        }

        // allows shifting the SDF outside the body:
        //const Real shift = h;
        static constexpr Real shift = 0;
        const Real ssdf = sdf[iy][ix] + shift; // negative outside
        if (ssdf > +2*h || ssdf < -2*h) continue; // no need to compute gradChi

        {
          const double distPx = (ix+1==_BS_? SDIST(ix+1,iy).s : sdf[iy][ix+1]) + shift;
          const double distMx = (ix  ==0   ? SDIST(ix-1,iy).s : sdf[iy][ix-1]) + shift;
          const double distPy = (iy+1==_BS_? SDIST(ix,iy+1).s : sdf[iy+1][ix]) + shift;
          const double distMy = (iy  ==0   ? SDIST(ix,iy-1).s : sdf[iy-1][ix]) + shift;

          const auto HplusX = std::fabs(distPx)<EPS? (double).5 :(distPx<0?0:1);
          const auto HminuX = std::fabs(distMx)<EPS? (double).5 :(distMx<0?0:1);
          const auto HplusY = std::fabs(distPy)<EPS? (double).5 :(distPy<0?0:1);
          const auto HminuY = std::fabs(distMy)<EPS? (double).5 :(distMy<0?0:1);

          const double gradUX= i2h*(distPx-distMx), gradUY= i2h*(distPy-distMy);
          const double gradHX=     (HplusX-HminuX), gradHY=     (HplusY-HminuY);

          const double gradUSq = gradUX * gradUX + gradUY * gradUY;
          const double denum = 1 / ( gradUSq<EPS ? EPS : gradUSq );
          //fac is 1/2h of gradH times h^2 to make \int_v D*gradU = \int_s norm
          const double D = fac*(gradHX*gradUX + gradHY*gradUY) * denum;
          o.write(ix, iy, D, gradUX, gradUY);
          //if(D>0) { //
          //  const double norx = -D*gradUX, nory = -D*gradUY;
          //  assert(std::sqrt(norx*norx + nory*nory) <= 1);
          //  udefoutflow += udef[iy][ix][0]*norx + udef[iy][ix][1]*nory;
          //  udefoutnorm += norx*norx + nory*nory;
          //}
        }
      }
    }
  }

  if(_m > EPS) {
    shape->centerOfMass[0] += _x/_m;
    shape->centerOfMass[1] += _y/_m;
    shape->M = _m;
  } else printf("PutObjectsOnGrid _m is too small!\n");

  for (auto & o : OBLOCK) if(o not_eq nullptr) o->allocate_surface();

  /*
  const Real outflowCorr = udefoutflow / std::max(udefoutnorm, EPS);
  double udefoutpost = 0;
  #pragma omp parallel for schedule(dynamic, 1) reduction(+ : udefoutpost)
  for (size_t j=0; j < Nblocks; ++j)
  {
    if(OBLOCK[chiInfo[j].blockID] == nullptr) continue; //obst not in block
    ObstacleBlock& o = * OBLOCK[chiInfo[j].blockID];
    UDEFMAT & __restrict__ udef = o.udef;

    for(size_t i=0; i < o.n_surfPoints; ++i)
    {
      const int ix = o.surface[i]->ix, iy = o.surface[i]->iy;
      const Real norx = o.surface[i]->dchidx, nory = o.surface[i]->dchidy;
      udef[iy][ix][0] -= outflowCorr * norx; //h^2 already premultiplied
      udef[iy][ix][1] -= outflowCorr * nory; //    in dchidx/dchidy
      udefoutpost += udef[iy][ix][0]*norx + udef[iy][ix][1]*nory;
    }
  }
  //printf("udefoutflow %e udefoutpost %e \n", udefoutflow,udefoutpost);
  */
}

void PutObjectsOnGrid::putObjectVelOnGrid(Shape * const shape) const
{
  const std::vector<ObstacleBlock*>& OBLOCK = shape->obstacleBlocks;
  //const Real h = sim.getH();
  //const double u_s = shape->u, v_s = shape->v, omega_s = shape->omega;
  //const double Cx = shape->centerOfMass[0], Cy = shape->centerOfMass[1];

  #pragma omp parallel for schedule(dynamic)
  for (size_t i=0; i < Nblocks; i++)
  {
    if(OBLOCK[uDefInfo[i].blockID] == nullptr) continue; //obst not in block
    //using UDEFMAT = Real[VectorBlock::sizeY][VectorBlock::sizeX][2];
    //using CHI_MAT = Real[VectorBlock::sizeY][VectorBlock::sizeX];

    const UDEFMAT & __restrict__ udef = OBLOCK[uDefInfo[i].blockID]->udef;
    const CHI_MAT & __restrict__ chi  = OBLOCK[uDefInfo[i].blockID]->chi;
    auto & __restrict__ UDEF = *(VectorBlock*)uDefInfo[i].ptrBlock; // dest
    //const ScalarBlock&__restrict__ TMP  = *(ScalarBlock*) tmpInfo[i].ptrBlock;
    const ScalarBlock&__restrict__ CHI  = *(ScalarBlock*) chiInfo[i].ptrBlock;

    for(int iy=0; iy<VectorBlock::sizeY; iy++)
    for(int ix=0; ix<VectorBlock::sizeX; ix++)
    {
      if( chi[iy][ix] < CHI(ix,iy).s || chi[iy][ix] <= 0) continue;
      Real p[2]; uDefInfo[i].pos(p, ix, iy);
      UDEF(ix, iy).u[0] += udef[iy][ix][0];
      UDEF(ix, iy).u[1] += udef[iy][ix][1];
    }
    //if (TMP(ix,iy).s > -3*h) //( chi[iy][ix] > 0 )
    //{ //plus equal in case of overlapping objects
    //  Real p[2]; uDefInfo[i].pos(p, ix, iy);
    //  UDEF(ix,iy).u[0] += u_s - omega_s*(p[1]-Cy) + udef[iy][ix][0];
    //  UDEF(ix,iy).u[1] += v_s + omega_s*(p[0]-Cx) + udef[iy][ix][1];
    //}
  }
}

void PutObjectsOnGrid::operator()(const double dt)
{
  // TODO I NEED SIGNED DISTANCE PER OBSTACLE
  // 0) clear chi^t and udef^t
  sim.startProfiler("ObjGrid_clear");
  #pragma omp parallel for schedule(static)
  for (size_t i=0; i < Nblocks; i++) {
    ( (ScalarBlock*)   chiInfo[i].ptrBlock )->clear();
    ( (ScalarBlock*)   tmpInfo[i].ptrBlock )->set(-1);
    ( (VectorBlock*)  uDefInfo[i].ptrBlock )->clear();
    ( (ScalarBlock*)invRhoInfo[i].ptrBlock )->set(1);
  }
  sim.stopProfiler();


  // 1) update objects' position (advect)
  sim.startProfiler("Obj_move");
  int nSum[2] = {0, 0}; double uSum[2] = {0, 0};
  for(Shape * const shape : sim.shapes) shape->updateLabVelocity(nSum, uSum);
  if(nSum[0]>0) sim.uinfx = uSum[0]/nSum[0];
  if(nSum[1]>0) sim.uinfy = uSum[1]/nSum[1];

  for(Shape * const shape : sim.shapes)
  {
    shape->updatePosition(dt);
    double p[2] = {0,0};
    shape->getCentroid(p);
    const auto& extent = sim.extents;
    if (p[0]<0 || p[0]>extent[0] || p[1]<0 || p[1]>extent[1]) {
      printf("Body out of domain [0,%f]x[0,%f] CM:[%e,%e]\n",
        extent[0], extent[1], p[0], p[1]);
      exit(0);
    }
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
    // put actual vel on the object vel grid
    //if(sim.bStaggeredGrid) putObjectVelOnGridStaggered(shape); else
    putObjectVelOnGrid(shape);
  }
  sim.stopProfiler();
}
