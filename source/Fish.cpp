//
//  IF2D_FishOperator.cpp
//  IF2D_ROCKS
//
//  Created by Wim van Rees on 17/11/14.
//
//


#include <chrono>
#include "Fish.h"
#include "FishLibrary.h"
#include <array>
#include <cassert>
#include <cmath>

void Fish::create(const vector<BlockInfo>& vInfo)
{
  // clear deformation velocities
  for(auto & entry : obstacleBlocks) delete entry.second;
  obstacleBlocks.clear();
  // STRATEGY
  // 1. update midline
  // 2. integrate to find CoM, angular velocity, etc
  // 3. shift midline to CoM frame: zero internal linear momentum and angular momentum

  // 4. split the fish into segments (according to s)
  // 5. rotate the segments to computational frame (comp CoM and angle)
  // 6. for each Block in the domain, find those segments that intersect it
  // 7. for each of those blocks, allocate a DeformingObstacleBlock

  // 8. put the 2D shape on the grid: SDF-P2M for sdf, normal P2M for udef
  // 9. create the Chi out of the SDF
  // 10. correct deformation velocity to nullify momenta

  assert(myFish!=nullptr);
  // 1.
  myFish->computeMidline(sim.time, sim.dt);
  //myFish->computeSurface();

  // 2. and 3.
  // returns area, CoM_internal, vCoM_internal:
  area_internal = myFish->integrateLinearMomentum(CoM_internal, vCoM_internal);
  assert(area_internal > std::numeric_limits<Real>::epsilon());
  // takes CoM_internal, vCoM_internal, puts CoM in 0 and nullifies lin mom:
  myFish->changeToCoMFrameLinear(CoM_internal, vCoM_internal);
  angvel_internal_prev = angvel_internal;
  // returns mom of intertia and angvel:
  J_internal = myFish->integrateAngularMomentum(angvel_internal);
  // rotates fish midline to current angle and removes angular moment:
  myFish->changeToCoMFrameAngular(theta_internal, angvel_internal);
  #ifndef NDEBUG
  {
    Real dummy_CoM_internal[2], dummy_vCoM_internal[2], dummy_angvel_internal;
    // check that things are zero
    const Real area_internal_check =
    myFish->integrateLinearMomentum(dummy_CoM_internal, dummy_vCoM_internal);
    myFish->integrateAngularMomentum(dummy_angvel_internal);
    const Real EPS = 10*std::numeric_limits<Real>::epsilon();
    assert(std::fabs(dummy_CoM_internal[0])<EPS);
    assert(std::fabs(dummy_CoM_internal[1])<EPS);
    assert(std::fabs(myFish->linMom[0])<EPS);
    assert(std::fabs(myFish->linMom[1])<EPS);
    assert(std::fabs(myFish->angMom)<EPS);
    assert(std::fabs(area_internal - area_internal_check) < EPS);
  }
  #endif
  //myFish->surfaceToCOMFrame(theta_internal,CoM_internal);

  //- VolumeSegment_OBB's volume cannot be zero
  //- therefore no VolumeSegment_OBB can be only occupied by extension midline
  //  points (which have width and height = 0)
  //- performance of create seems to decrease if VolumeSegment_OBB are bigger
  //- this is the smallest number of VolumeSegment_OBB (Nsegments) and points in
  //  the midline (Nm) to ensure at least one non ext. point inside all segments
  const int Nsegments = (myFish->Nm-1)/8;
  const int Nm = myFish->Nm;
  assert((Nm-1)%Nsegments==0);

  vector<AreaSegment*> vSegments(Nsegments, nullptr);
  #pragma omp parallel for
  for(int i=0; i<Nsegments; ++i) {
    const int next_idx = (i+1)*(Nm-1)/Nsegments;
    const int idx = i * (Nm-1)/Nsegments;
    // find bounding box based on this
    Real bbox[2][2] = {{1e9, -1e9}, {1e9, -1e9}};
    for(int ss=idx; ss<=next_idx; ++ss) {
      const Real xBnd[2]={myFish->rX[ss] -myFish->norX[ss]*myFish->width[ss],
                          myFish->rX[ss] +myFish->norX[ss]*myFish->width[ss]};
      const Real yBnd[2]={myFish->rY[ss] -myFish->norY[ss]*myFish->width[ss],
                          myFish->rY[ss] +myFish->norY[ss]*myFish->width[ss]};
      const Real maxX=std::max(xBnd[0],xBnd[1]);
      const Real minX=std::min(xBnd[0],xBnd[1]);
      const Real maxY=std::max(yBnd[0],yBnd[1]);
      const Real minY=std::min(yBnd[0],yBnd[1]);
      bbox[0][0] = std::min(bbox[0][0], minX);
      bbox[0][1] = std::max(bbox[0][1], maxX);
      bbox[1][0] = std::min(bbox[1][0], minY);
      bbox[1][1] = std::max(bbox[1][1], maxY);
    }

    // 4.
    const Real buf = 2*vInfo[0].h_gridpoint; //two points on each side
    //const Real safe_distance = info.h_gridpoint; // one point on each side
    AreaSegment*const tAS = new AreaSegment(make_pair(idx,next_idx), bbox, buf);
    //5.
    tAS->changeToComputationalFrame(center, orientation);
    vSegments[i] = tAS;
  }

  map<int,vector<AreaSegment*>> segmentsPerBlock;
  for(size_t i=0; i<vInfo.size(); ++i) {
    const BlockInfo & info = vInfo[i];
    Real pStart[2], pEnd[2];
    info.pos(pStart, 0, 0);
    info.pos(pEnd, FluidBlock::sizeX-1, FluidBlock::sizeY-1);

    // 6.
    for(size_t s=0; s<vSegments.size(); ++s)
      if(vSegments[s]->isIntersectingWithAABB(pStart,pEnd))
        segmentsPerBlock[info.blockID].push_back(vSegments[s]);

    // 7.
    // allocate new blocks if necessary
    if(segmentsPerBlock.find(info.blockID) != segmentsPerBlock.end()) {
      assert(obstacleBlocks.find(info.blockID) == obstacleBlocks.end());
      ObstacleBlock * const block = new ObstacleBlock();
      assert(block not_eq nullptr);
      obstacleBlocks[info.blockID] = block;
      block->clear();
    }
  }
  assert(not segmentsPerBlock.empty());
  assert(segmentsPerBlock.size() == obstacleBlocks.size());

  // 8.
  #pragma omp parallel
  {
    const PutFishOnBlocks putfish(*myFish, center, orientation);

    #pragma omp for schedule(dynamic)
    for(size_t i=0; i<vInfo.size(); i++) {
      const BlockInfo& info = vInfo[i];
      FluidBlock& b = *(FluidBlock*)info.ptrBlock;
      const auto pos = segmentsPerBlock.find(info.blockID);

      if(pos != segmentsPerBlock.end()) {
        for(int iy=0; iy<FluidBlock::sizeY; ++iy)
        for(int ix=0; ix<FluidBlock::sizeX; ++ix)
          b(ix,iy).tmpU = 0; //this will be accessed with plus equal

        assert(obstacleBlocks.find(info.blockID) != obstacleBlocks.end());
        ObstacleBlock*const block = obstacleBlocks.find(info.blockID)->second;
        putfish(info, b, block, pos->second);
      }
    }
  }

  // clear vSegments
  for(auto & entry : vSegments) { assert(entry not_eq nullptr); delete entry; }
  // 9.
  //here chi contains signed distance from only this fish, rest of chi field is in tmp
  //obstBlock->chi is the squared signed distance
  {
    const PutFishOnBlocks_Finalize finalize = PutFishOnBlocks_Finalize();
    compute(finalize, vInfo);
  }

  // 10.
  removeMoments(vInfo);
  //myFish->surfaceToComputationalFrame(orientation, centerOfMass);
  for(auto & o : obstacleBlocks) o.second->allocate_surface();
}

void Fish::updatePosition(double dt) {
  // update position and angles
  Shape::updatePosition(dt);
  theta_internal -= dt*angvel_internal; // negative: we subtracted this angvel
}
void Fish::resetAll() {
  CoM_internal[0] = 0; CoM_internal[1] = 0;
  vCoM_internal[0] = 0; vCoM_internal[1] = 0;
  theta_internal = 0; angvel_internal = 0; angvel_internal_prev = 0;
  Shape::resetAll();
  myFish->resetAll();
}

Fish::~Fish() {
  if(myFish not_eq nullptr) {
    delete myFish;
    myFish = nullptr;
  }
}

#if 0
void Fish::apply_pid_corrections()
{

  if (followX > 0 && followY > 0) //then i control the position
  {
    assert(not bCorrectTrajectory);
    //const double velx_tot = Uinf[0] - transVel[0];
    //const double vely_tot = Uinf[1] - transVel[1];
    const double AngDiff  = _2Dangle;//std::atan2(vely_tot,velx_tot);

    // Control posDiffs
    const double xDiff = (position[0] - followX)/length;
    const double yDiff = (position[1] - followY)/length;
    const double absDY = std::fabs(yDiff);
    const double velAbsDY = yDiff>0 ? transVel[1]/length : -transVel[1]/length;
    const double velDAvg = AngDiff-adjTh + dt*angVel[2];

    adjTh = (1.-dt) * adjTh + dt * AngDiff;
    adjDy = (1.-dt) * adjDy + dt * yDiff;

    //If angle is positive: positive curvature only if Dy<0 (must go up)
    //If angle is negative: negative curvature only if Dy>0 (must go down)
    //const Real INST = (AngDiff*angVel[2]>0 && yDiff*AngDiff<0) ? AngDiff*std::fabs(yDiff*angVel[2]) : 0;
    const double PROP = (adjTh  *yDiff<0) ?   adjTh*absDY : 0;
    const double INST = (AngDiff*yDiff<0) ? AngDiff*absDY : 0;

    //zero also the derivatives when appropriate
    const double f1 = std::fabs(PROP)>2e-16 ? 20 : 0;
    const double f2 = std::fabs(INST)>2e-16 ? 50 : 0, f3=1;

    // Linearly increase (or decrease) amplitude to 1.2X (decrease to 0.8X)
    //(experiments observed 1.2X increase in amplitude when swimming faster)
    //if fish falls back 1 body length. Beyond that, will still increase but dunno if will work
    const double ampFac = f3*xDiff + 1.0;
    const double ampVel = f3*transVel[0]/length;

    const double curv1fac = f1*PROP;
    const double curv1vel = f1*(velAbsDY*adjTh   + absDY*velDAvg);
    const double curv2fac = f2*INST;
    const double curv2vel = f2*(velAbsDY*AngDiff + absDY*angVel[2]);
                //const Real vPID = velAbsDY*(f1*adjTh + f2*AngDiff) + absDY*(f1*velDAvg+f2*angVel[2]);
                //const Real PID = f1*PROP + f2*INST;
    if(!rank) printf("%f\t f1: %f %f\t f2: %f %f\t f3: %f %f\n", time,
      curv1fac, curv1vel, curv2fac, curv2vel, ampFac, ampVel);
    myFish->_correctTrajectory(curv1fac+curv2fac, curv1vel+curv2vel, time, dt);
    myFish->_correctAmplitude(ampFac, ampVel, time, dt);
  }
}
#endif
