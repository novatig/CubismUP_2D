//
//  CubismUP_2D
//  Copyright (c) 2018 CSE-Lab, ETH Zurich, Switzerland.
//  Distributed under the terms of the MIT license.
//
//  Created by Guido Novati (novatig@ethz.ch).
//


#include "StefanFish.h"
#include "FishLibrary.h"
#include "FishUtilities.h"
#include <sstream>

using namespace cubism;

class CurvatureFish : public FishData
{
 protected:
  Real * const rK;
  Real * const vK;
  Real * const rC;
  Real * const vC;
  Real * const rB;
  Real * const vB;
  Real * const rA;
  Real * const vA;
  bool useFollowXY_PID = false;
  Real valPID = 0;
  Real velPID = 0;
  Real d_Tp = 0;
  Schedulers::ParameterSchedulerVector<6> curvScheduler;
  Schedulers::ParameterSchedulerLearnWave<7> baseScheduler;
  Schedulers::ParameterSchedulerVector<6> adjustScheduler;
 public:

  CurvatureFish(Real L, Real T, Real phi, Real _h, Real _A)
  : FishData(L, T, phi, _h, _A),
    rK(_alloc(Nm)),vK(_alloc(Nm)), rC(_alloc(Nm)),vC(_alloc(Nm)),
    rB(_alloc(Nm)),vB(_alloc(Nm)), rA(_alloc(Nm)),vA(_alloc(Nm)) {
      _computeWidth();
      writeMidline2File(0, "initialCheck");
    }
  void resetAll() override;

  void _correctTrajectory(const Real dtheta, const Real vtheta,
                          const Real time, Real dt);

  void _correctAmplitude(      Real periodFac, Real periodVel,
                         const Real lastTime, const Real time,
                         const Real dt);

  void execute(const Real time, const Real l_tnext, const std::vector<double>& input);

  ~CurvatureFish() {
    _dealloc(rK); _dealloc(vK); _dealloc(rC); _dealloc(vC);
    _dealloc(rB); _dealloc(vB); _dealloc(rA); _dealloc(vA);
  }

  void computeMidline(const Real time, const Real dt) override;
  Real _width(const Real s, const Real L) override
  {
    const Real sb=.04*length, st=.95*length, wt=.01*length, wh=.04*length;
    if(s<0 or s>L) return 0;
    return (s<sb ? std::sqrt(2*wh*s -s*s) :
           (s<st ? wh-(wh-wt)*std::pow((s-sb)/(st-sb),1) : // pow(.,2) is 3D
           (wt * (L-s)/(L-st))));
  }
};

void CurvatureFish::resetAll() {
  valPID = 0;
  velPID = 0;
  useFollowXY_PID = false;
  curvScheduler.resetAll();
  baseScheduler.resetAll();
  adjustScheduler.resetAll();
  FishData::resetAll();
}

void CurvatureFish::_correctTrajectory(const Real dtheta, const Real vtheta,
                                       const Real t, Real dt)
{
  valPID = dtheta;
  velPID = vtheta;
  dt = std::max(std::numeric_limits<Real>::epsilon(),dt);
  std::array<Real, 6> tmp_curv;
  tmp_curv.fill(dtheta);
  //adjustScheduler.transition(time,time,time+2*dt,tmp_curv, true);
  adjustScheduler.transition(t, t-2*dt, t+2*dt, tmp_curv, true);
}


void CurvatureFish::_correctAmplitude(Real periodFac, Real periodVel,
                                      const Real lastTime, const Real time,
                                      const Real dt)
{
  assert(periodFac>0 && periodFac<2); // would be crazy
  if(periodFac <= 0) { periodFac=0; periodVel=0; }

  const Real lastArg = (lastTime-time0)/l_Tp + timeshift;
  time0 = lastTime;
  timeshift = lastArg;
  // so that new arg is only constant (prev arg) + dt / l_Tp
  // with the new l_Tp:
  l_Tp = Tperiod * periodFac;
  d_Tp = Tperiod * periodVel;
  useFollowXY_PID = true;
}

void CurvatureFish::execute(const Real t, const Real lTact,
                            const std::vector<double>&a)
{
  assert(t >= lTact);
  if (a.size()>1) {
    baseScheduler.Turn(a[0], lTact);
    //first, shift time to  previous turn node
    timeshift += (lTact-time0)/l_Tp;
    time0 = lTact;
    l_Tp = Tperiod*(1+a[1]);
  } else if (a.size()>0) {
    printf("Turning by %g at time %g with period %g.\n", a[0], t, lTact);
    baseScheduler.Turn(a[0], lTact);
  }
}

void CurvatureFish::computeMidline(const Real time, const Real dt)
{
  const std::array<Real ,6> curvature_points = { (Real)0, (Real).15*length,
    (Real).4*length, (Real).65*length, (Real).9*length, length
  };
  const std::array<Real,7> baseline_points = {(Real)-.5, (Real)-.25, (Real)0,
    (Real).25, (Real).5, (Real).75, (Real)1};
  const std::array<Real ,6> curvature_values = {
      (Real)0.82014/length, (Real)1.46515/length, (Real)2.57136/length,
      (Real)3.75425/length, (Real)5.09147/length, (Real)5.70449/length
  };
  #if 1
  const std::array<Real,6> curvature_zeros = std::array<Real, 6>();
  curvScheduler.transition(0,0, Tperiod, curvature_zeros, curvature_values);
  #else // no rampup for debug
  curvScheduler.transition(time,0,Tperiod,curvature_values,curvature_values);
  #endif
  //const std::array<Real ,6> curvature_values = std::array<Real, 6>();
  //
  // query the schedulers for current values
  //std::stringstream curvCout;
  //curvCout << "Profile t:"<<time<<" t0:"<<baseScheduler.t0;
  //for(int i=0; i<7; ++i) curvCout <<baseScheduler.parameters_t0[i] <<" ";
  //std::cout << curvCout.str() << std::endl;
    curvScheduler.gimmeValues(time,            curvature_points, Nm,rS, rC,vC);
    baseScheduler.gimmeValues(time,l_Tp,length, baseline_points, Nm,rS, rB,vB);
  adjustScheduler.gimmeValues(time,            curvature_points, Nm,rS, rA,vA);
  if(useFollowXY_PID) {
    const Real _vA = velPID, _rA = valPID;
    //#pragma omp parallel for schedule(static)
    for(int i=0; i<Nm; ++i) {
      const Real darg = 2*M_PI*(1/l_Tp - (time-time0)*d_Tp/l_Tp/l_Tp);
      const Real arg  = 2*M_PI*((time-time0)/l_Tp +timeshift
                                -rS[i]/length/waveLength) + M_PI*phaseShift;
      rK[i] = amplitudeFactor* rC[i]*(std::sin(arg)     + rB[i] + _rA);
      vK[i] = amplitudeFactor*(vC[i]*(std::sin(arg)     + rB[i] + _rA)
                              +rC[i]*(std::cos(arg)*darg+ vB[i] + _vA));
    }
  } else {
    //#pragma omp parallel for schedule(static)
    for(int i=0; i<Nm; ++i) {
      const Real darg = 2*M_PI/l_Tp;
      const Real arg  = 2*M_PI*((time-time0)/l_Tp +timeshift
                                -rS[i]/length/waveLength) + M_PI*phaseShift;
      rK[i] = amplitudeFactor*  rC[i]*(std::sin(arg)      + rB[i] + rA[i]);
      vK[i] = amplitudeFactor* (vC[i]*(std::sin(arg)      + rB[i] + rA[i])
                              + rC[i]*(std::cos(arg)*darg + vB[i] + vA[i]));
      assert(not std::isnan(rK[i]));
      assert(not std::isinf(rK[i]));
      assert(not std::isnan(vK[i]));
      assert(not std::isinf(vK[i]));
    }
  }

  // solve frenet to compute midline parameters
  IF2D_Frenet2D::solve(Nm, rS, rK,vK, rX,rY, vX,vY, norX,norY, vNorX,vNorY);
  #if 0
   {
    FILE * f = fopen("stefan_profile","w");
    for(int i=0;i<Nm;++i)
      fprintf(f,"%d %g %g %g %g %g %g %g %g %g\n",
        i,rS[i],rX[i],rY[i],vX[i],vY[i],
        vNorX[i],vNorY[i],width[i],height[i]);
    fclose(f);
   }
  #endif
}

void StefanFish::resetAll() {
  adjTh = 0; adjDy = 0;
  lastTact = 0; lastCurv = 0; oldrCurv = 0;
  Fish::resetAll();
}

StefanFish::StefanFish(SimulationData&s, ArgumentParser&p, double C[2]):
 Fish(s,p,C),
 followX(p("-followX").asDouble(-1)), followY(p("-followY").asDouble(-1)),
 bCorrectTrajectory(p("-pid").asInt(0))
{
 #if 0
  // parse tau
  tau = parser("-tau").asDouble(1.0);
  // parse curvature controlpoint values
  curvature_values[0] = parser("-k1").asDouble(0.82014);
  curvature_values[1] = parser("-k2").asDouble(1.46515);
  curvature_values[2] = parser("-k3").asDouble(2.57136);
  curvature_values[3] = parser("-k4").asDouble(3.75425);
  curvature_values[4] = parser("-k5").asDouble(5.09147);
  curvature_values[5] = parser("-k6").asDouble(5.70449);
  // if nonzero && Learnfreq<0 your fish is gonna keep turning
  baseline_values[0] = parser("-b1").asDouble(0.0);
  baseline_values[1] = parser("-b2").asDouble(0.0);
  baseline_values[2] = parser("-b3").asDouble(0.0);
  baseline_values[3] = parser("-b4").asDouble(0.0);
  baseline_values[4] = parser("-b5").asDouble(0.0);
  baseline_values[5] = parser("-b6").asDouble(0.0);
  // curvature points are distributed by default but can be overridden
  curvature_points[0] = parser("-pk1").asDouble(0.00)*length;
  curvature_points[1] = parser("-pk2").asDouble(0.15)*length;
  curvature_points[2] = parser("-pk3").asDouble(0.40)*length;
  curvature_points[3] = parser("-pk4").asDouble(0.65)*length;
  curvature_points[4] = parser("-pk5").asDouble(0.90)*length;
  curvature_points[5] = parser("-pk6").asDouble(1.00)*length;
  baseline_points[0] = parser("-pb1").asDouble(curvature_points[0]/length)*length;
  baseline_points[1] = parser("-pb2").asDouble(curvature_points[1]/length)*length;
  baseline_points[2] = parser("-pb3").asDouble(curvature_points[2]/length)*length;
  baseline_points[3] = parser("-pb4").asDouble(curvature_points[3]/length)*length;
  baseline_points[4] = parser("-pb5").asDouble(curvature_points[4]/length)*length;
  baseline_points[5] = parser("-pb6").asDouble(curvature_points[5]/length)*length;
  printf("created IF2D_StefanFish: xpos=%3.3f ypos=%3.3f angle=%3.3f L=%3.3f Tp=%3.3f tau=%3.3f phi=%3.3f\n",position[0],position[1],angle,length,Tperiod,tau,phaseShift);
  printf("curvature points: pk1=%3.3f pk2=%3.3f pk3=%3.3f pk4=%3.3f pk5=%3.3f pk6=%3.3f\n",curvature_points[0],curvature_points[1],curvature_points[2],curvature_points[3],curvature_points[4],curvature_points[5]);
  printf("curvature values (normalized to L=1): k1=%3.3f k2=%3.3f k3=%3.3f k4=%3.3f k5=%3.3f k6=%3.3f\n",curvature_values[0],curvature_values[1],curvature_values[2],curvature_values[3],curvature_values[4],curvature_values[5]);
  printf("baseline points: pb1=%3.3f pb2=%3.3f pb3=%3.3f pb4=%3.3f pb5=%3.3f pb6=%3.3f\n",baseline_points[0],baseline_points[1],baseline_points[2],baseline_points[3],baseline_points[4],baseline_points[5]);
  printf("baseline values (normalized to L=1): b1=%3.3f b2=%3.3f b3=%3.3f b4=%3.3f b5=%3.3f b6=%3.3f\n",baseline_values[0],baseline_values[1],baseline_values[2],baseline_values[3],baseline_values[4],baseline_values[5]);
  // make curvature dimensional for this length
  for(int i=0; i<6; ++i) curvature_values[i]/=length;
 #endif

  const Real ampFac = p("-amplitudeFactor").asDouble(1.0);
  myFish = new CurvatureFish(length, Tperiod, phaseShift, sim.getH(), ampFac);
  printf("CurvatureFish %d %f %f %f\n",myFish->Nm, length, Tperiod, phaseShift);
}

//static inline Real sgn(const Real val) { return (0 < val) - (val < 0); }
void StefanFish::create(const std::vector<BlockInfo>& vInfo)
{
  CurvatureFish* const cFish = dynamic_cast<CurvatureFish*>( myFish );
  if(cFish == nullptr) { printf("Someone touched my fish\n"); abort(); }

  if (bCorrectTrajectory) {
    assert(followX < 0 && followY < 0);
    const Real AngDiff  = std::atan2(-v, -u);
    adjTh = (Tperiod-sim.dt)/Tperiod * adjTh + sim.dt/Tperiod * AngDiff;
    const Real INST = (AngDiff*omega>0) ? AngDiff*std::fabs(omega)*Tperiod : 0;
    const Real PID  = 0.1*adjTh + 0.01*INST;
    if(not sim.muteAll && sim.dt>0) {
      std::ofstream filePID;
      std::stringstream ssF; ssF<<sim.path2file<<"/PID_"<<obstacleID<<".dat";
      filePID.open(ssF.str().c_str(), std::ios::app);
      filePID<<adjTh<<" "<<INST<<"\n";
    }
    cFish->_correctTrajectory(PID, 0, sim.time, sim.dt); // 2nd arg unused
  }

  if (followX > 0 && followY > 0) //then i control the position
  {
    assert(not bCorrectTrajectory);
    const double angDiff  = std::atan2(-v, -u), dt = sim.dt, time = sim.time;
    const double relU = u + sim.uinfx, relV = v + sim.uinfy;
    // Control pos diffs
    const double xDiff = (centerOfMass[0] - followX)/length;
    const double yDiff = (centerOfMass[1] - followY)/length;
    // derivatives of following 2 exponential averages:
    const double velDAavg = (angDiff-adjTh)/Tperiod + dt/Tperiod * omega;
    const double velDYavg = (  yDiff-adjDy)/Tperiod + dt/Tperiod * relV/length;

    adjTh = (1.0-dt/Tperiod) * adjTh + dt/Tperiod * angDiff;
    adjDy = (1.0-dt/Tperiod) * adjDy + dt/Tperiod *   yDiff;

    // integral (averaged) and proportional absolute DY and their derivative
    const double absPy = std::fabs(yDiff), absIy = std::fabs(adjDy);
    const double velAbsPy = yDiff>0 ? relV/length : -relV/length;
    const double velAbsIy = adjDy>0 ?    velDYavg : -   velDYavg;
    //If angle is positive: positive curvature only if Dy<0 (must go up)
    //If angle is negative: negative curvature only if Dy>0 (must go down)
    const double IangPdy = ( adjTh  * yDiff < 0)?   adjTh * absPy : 0;
    const double PangIdy = (angDiff * adjDy < 0)? angDiff * absIy : 0;
    const double IangIdy = ( adjTh  * adjDy < 0)?   adjTh * absIy : 0;

    // derivatives multiplied by 0 when term is inactive later:
    const double velIangPdy = velAbsPy * adjTh   + absPy * velDAavg;
    const double velPangIdy = velAbsIy * angDiff + absIy * omega;
    const double velIangIdy = velAbsIy * adjTh   + absIy * velDAavg;

    //zero also the derivatives when appropriate
    const double coefIangPdy =  adjTh  * yDiff < 0 ? 20 : 0;
    const double coefPangIdy = angDiff * adjDy < 0 ? 0 : 0;
    const double coefIangIdy =  adjTh  * adjDy < 0 ? 20 : 0;


    const double valIangPdy = coefIangPdy * IangPdy;
    const double difIangPdy = coefIangPdy * velIangPdy;
    const double valPangIdy = coefPangIdy * PangIdy;
    const double difPangIdy = coefPangIdy * velPangIdy;
    const double valIangIdy = coefIangIdy * IangIdy;
    const double difIangIdy = coefIangIdy * velIangIdy;
    const double periodFac = 1.0 - xDiff;
    const double periodVel =     - relU/length;

    if(not sim.muteAll && sim.dt>0) {
      std::ofstream filePID;
      std::stringstream ssF;
      ssF<<sim.path2file<<"/PID_"<<obstacleID<<".dat";
      filePID.open(ssF.str().c_str(), std::ios::app);
      filePID<<time<<" "<<valIangPdy<<" "<<difIangPdy
                   <<" "<<valPangIdy<<" "<<difPangIdy
                   <<" "<<valIangIdy<<" "<<difIangIdy
                   <<" "<<periodFac <<" "<<periodVel <<"\n";
    }
    const double totalTerm = valIangPdy + valPangIdy + valIangIdy;
    const double totalDiff = difIangPdy + difPangIdy + difIangIdy;
    cFish->_correctTrajectory(totalTerm, totalDiff, time, dt);
    cFish->_correctAmplitude(periodFac, periodVel, lastTime, time, dt);
  }

  lastTime = sim.time;
  Fish::create(vInfo);
}

void StefanFish::act(const Real lTact, const std::vector<double>& a) const
{
  CurvatureFish* const cFish = dynamic_cast<CurvatureFish*>( myFish );
  oldrCurv = lastCurv;
  lastCurv = a[0];
  if(a.size()>1) lastTact = a[1];
  cFish->execute(sim.time, lTact, a);
}

double StefanFish::getLearnTPeriod() const
{
  return myFish->l_Tp;
}

double StefanFish::getPhase(const double t) const
{
  const double Tp = myFish->l_Tp;
  const double T0 = myFish->time0;
  const double Ts = myFish->timeshift;
  const double arg  = 2*M_PI*((t-T0)/Tp +Ts) + M_PI*phaseShift;
  const double phase = std::fmod(arg, 2*M_PI);
  return (phase<0) ? 2*M_PI + phase : phase;
}

std::vector<double> StefanFish::state(Shape*const p) const
{
  std::vector<double> S(10,0);
  S[0] = ( center[0] - p->center[0] )/ length;
  S[1] = ( center[1] - p->center[1] )/ length;
  S[2] = getOrientation();
  S[3] = getPhase( sim.time );
  S[4] = getU() * Tperiod / length;
  S[5] = getV() * Tperiod / length;
  S[6] = getW() * Tperiod;
  S[7] = lastTact;
  S[8] = lastCurv;
  S[9] = oldrCurv;

  #ifndef STEFANS_SENSORS_STATE
    return S;
  #else
    S.resize(16);
    const Real h = sim.getH(), invh = 1/h;
    const auto holdsPoint = [&](const BlockInfo& I, const Real X, const Real Y)
    {
      Real MIN[2], MAX[2];
      I.pos(MIN, 0,0);
      I.pos(MAX, VectorBlock::sizeX-1, VectorBlock::sizeY-1);
      for(int i=0; i<2; ++i) {
          MIN[i] -= 0.5 * h; // pos returns cell centers
          MAX[i] += 0.5 * h; // we care about whole block
      }
      return X >= MIN[0] && Y >= MIN[1] && X <= MAX[0] && Y <= MAX[1];
    };

    // side of the head defined by position sb from function _width above ^^^
    int iHeadSide = 0;
    for(int i=0; i<myFish->Nm-1; ++i)
      if( myFish->rS[i] <= 0.04*length && myFish->rS[i+1] > 0.04*length )
        iHeadSide = i;
    assert(iHeadSide>0);

    std::array<Real,2> tipShear, lowShear, topShear;
    const std::vector<cubism::BlockInfo>& velInfo = sim.vel->getBlocksInfo();
    #pragma omp parallel for schedule(dynamic)
    for(size_t i=0; i<velInfo.size(); ++i)
    {
      {
        const auto &DU = myFish->upperSkin, &DL = myFish->lowerSkin;
        const Real skinX=DU.xSurf[0], normX=(DU.normXSurf[0]+DL.normXSurf[0])/2;
        const Real skinY=DU.ySurf[0], normY=(DU.normYSurf[0]+DL.normYSurf[0])/2;
        const Real sensX = skinX + 2*h * normX, sensY = skinY + 2*h * normY;
        if( not holdsPoint(velInfo[i], sensX, sensY) ) continue;

        const ObstacleBlock*const o = obstacleBlocks[velInfo[i].blockID];
        if (o == nullptr) {
          printf("ABORT: sensor point is outside allocated obstacle blocks\n");
          fflush(0); abort();
        }
        Real org[2]; velInfo[i].pos(org, 0, 0);
        const int indx = (int) std::round((sensX - org[0])*invh);
        const int indy = (int) std::round((sensY - org[1])*invh);
        const int ix = std::min( std::max(0, indx), VectorBlock::sizeX-1);
        const int iy = std::min( std::max(0, indy), VectorBlock::sizeY-1);
        const VectorBlock& b = * (const VectorBlock*) velInfo[i].ptrBlock;
        const auto&__restrict__ udef = obstacleBlocks[velInfo[i].blockID]->udef;
        const Real uSkin = u - omega*(skinX-centerOfMass[0]) + udef[iy][ix][0];
        const Real vSkin = v + omega*(skinY-centerOfMass[1]) + udef[iy][ix][1];
        tipShear[0] = (b(ix, iy).u[0] - uSkin) * invh/2;
        tipShear[1] = (b(ix, iy).u[1] - vSkin) * invh/2;

        printf("tip sensor:[%f %f]->[%f %f] ind:[%d %d] val:%f %f\n",
        skinX, skinY, org[0], org[1], ix, iy, tipShear[0], tipShear[1]);
      }
      for(int a = 0; a<2; ++a)
      {
        const auto& D = a==0? myFish->upperSkin : myFish->lowerSkin;
        const Real skinX = D.midX[iHeadSide], normX = D.normXSurf[iHeadSide];
        const Real skinY = D.midY[iHeadSide], normY = D.normYSurf[iHeadSide];
        const Real sensX = skinX + 2*h * normX, sensY = skinY + 2*h * normY;
        if( not holdsPoint(velInfo[i], sensX, sensY) ) continue;

        const ObstacleBlock*const o = obstacleBlocks[velInfo[i].blockID];
        if (o == nullptr) {
          printf("ABORT: sensor point is outside allocated obstacle blocks\n");
          fflush(0); abort();
        }
        Real org[2]; velInfo[i].pos(org, 0, 0);
        const int indx = (int) std::round((sensX - org[0])*invh);
        const int indy = (int) std::round((sensY - org[1])*invh);
        const int ix = std::min( std::max(0, indx), VectorBlock::sizeX-1);
        const int iy = std::min( std::max(0, indy), VectorBlock::sizeY-1);
        const VectorBlock& b = * (const VectorBlock*) velInfo[i].ptrBlock;
        const auto&__restrict__ udef = obstacleBlocks[velInfo[i].blockID]->udef;
        const Real uSkin = u - omega*(skinX-centerOfMass[0]) + udef[iy][ix][0];
        const Real vSkin = v + omega*(skinY-centerOfMass[1]) + udef[iy][ix][1];
        const Real shearX = (b(ix, iy).u[0] - uSkin) * invh/2;
        const Real shearY = (b(ix, iy).u[1] - vSkin) * invh/2;
        const Real dX = D.xSurf[iHeadSide+1] - D.xSurf[iHeadSide];
        const Real dY = D.ySurf[iHeadSide+1] - D.ySurf[iHeadSide];
        const Real proj = dX * normX - dY * normY;
        const Real tangX = proj>0?  normX : -normX;
        const Real tangY = proj>0? -normY :  normY;
        (a==0? topShear[0] : lowShear[0]) = shearX * normX + shearY * normY;
        (a==0? topShear[1] : lowShear[1]) = shearX * tangX + shearY * tangY;
        if(a==0)
          printf("top sensor:[%f %f]->[%f %f] ind:[%d %d] val:%f %f\n",
          skinX, skinY, org[0], org[1], ix, iy, topShear[0], topShear[1]);
        else
          printf("bot sensor:[%f %f]->[%f %f] ind:[%d %d] val:%f %f\n",
          skinX, skinY, org[0], org[1], ix, iy, lowShear[0], lowShear[1]);
      }
    }

    S[10] = tipShear[0] * Tperiod / length;
    S[11] = tipShear[1] * Tperiod / length;
    S[12] = lowShear[0] * Tperiod / length;
    S[13] = lowShear[1] * Tperiod / length;
    S[14] = topShear[0] * Tperiod / length;
    S[15] = topShear[1] * Tperiod / length;

    return S;
  #endif
}
