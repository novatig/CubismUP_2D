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
 public:
  // PID controller of body curvature:
  Real curv_PID_fac = 0;
  Real curv_PID_dif = 0;
  // exponential averages:
  Real avgDeltaY = 0;
  Real avgDangle = 0;
  Real avgAngVel = 0;
  // stored past action for RL state:
  Real lastTact = 0;
  Real lastCurv = 0;
  Real oldrCurv = 0;
  // quantities needed to correctly control the speed of the midline maneuvers:
  Real periodPIDval = Tperiod;
  Real periodPIDdif = 0;
  bool TperiodPID = false;
  // quantities needed for rl:
  Real time0 = 0;
  Real timeshift = 0;
  // aux quantities for PID controllers:
  Real lastTime = 0;
  Real lastAvel = 0;

 protected:
  Real * const rK;
  Real * const vK;
  Real * const rC;
  Real * const vC;
  Real * const rB;
  Real * const vB;

  // next scheduler is used to ramp-up the curvature from 0 during first period:
  Schedulers::ParameterSchedulerVector<6> curvatureScheduler;
  // next scheduler is used for midline-bending control points for RL:
  Schedulers::ParameterSchedulerLearnWave<7> rlBendingScheduler;
 public:

  CurvatureFish(Real L, Real T, Real phi, Real _h, Real _A)
  : FishData(L, T, phi, _h, _A),   rK(_alloc(Nm)),vK(_alloc(Nm)),
    rC(_alloc(Nm)),vC(_alloc(Nm)), rB(_alloc(Nm)),vB(_alloc(Nm)) {
      _computeWidth();
      writeMidline2File(0, "initialCheck");
    }

  void resetAll() override {
    curv_PID_fac = 0;
    curv_PID_dif = 0;
    avgDeltaY = 0;
    avgDangle = 0;
    avgAngVel = 0;
    lastTact = 0;
    lastCurv = 0;
    oldrCurv = 0;
    periodPIDval = Tperiod;
    periodPIDdif = 0;
    TperiodPID = false;
    timeshift = 0;
    lastTime = 0;
    lastAvel = 0;
    time0 = 0;
    curvatureScheduler.resetAll();
    rlBendingScheduler.resetAll();
    FishData::resetAll();
  }

  void correctTrajectory(const Real dtheta, const Real vtheta,
                                         const Real t, const Real dt)
  {
    curv_PID_fac = dtheta;
    curv_PID_dif = vtheta;
  }

  void correctTailPeriod(const Real periodFac,const Real periodVel,
                                        const Real t, const Real dt)
  {
    assert(periodFac>0 && periodFac<2); // would be crazy

    const Real lastArg = (lastTime-time0)/periodPIDval + timeshift;
    time0 = lastTime;
    timeshift = lastArg;
    // so that new arg is only constant (prev arg) + dt / periodPIDval
    // with the new l_Tp:
    periodPIDval = Tperiod * periodFac;
    periodPIDdif = Tperiod * periodVel;
    lastTime = t;
    TperiodPID = true;
  }

// Execute takes as arguments the current simulation time and the time
// the RL action should have actually started. This is important for the midline
// bending because it relies on matching the splines with the half period of
// the sinusoidal describing the swimming motion (in order to exactly amplify
// or dampen the undulation). Therefore, for Tp=1, t_rlAction might be K * 0.5
// while t_current would be K * 0.5 plus a fraction of the timestep. This
// because the new RL discrete step is detected as soon as t_current>=t_rlAction
  void execute(const Real t_current, const Real t_rlAction,
                              const std::vector<double>&a)
  {
    assert(t_current >= t_rlAction);
    oldrCurv = lastCurv; // store action
    lastCurv = a[0]; // store action

    rlBendingScheduler.Turn(a[0], t_rlAction);
    printf("Turning by %g at time %g with period %g.\n",
           a[0], t_current, t_rlAction);

    if (a.size()>1) // also modify the swimming period
    {
      lastTact = a[1]; // store action
      // this is arg of sinusoidal before change-of-Tp begins:
      const Real lastArg = (t_rlAction - time0)/periodPIDval + timeshift;
      // make sure change-of-Tp affects only body config for future times:
      timeshift = lastArg; // add phase shift that accounts for current body shape
      time0 = t_rlAction; // shift time axis of undulation
      periodPIDval = Tperiod * (1 + a[1]); // actually change period
      periodPIDdif = 0; // constant periodPIDval between infrequent actions
    }
  }

  ~CurvatureFish() {
    _dealloc(rK); _dealloc(vK); _dealloc(rC); _dealloc(vC);
    _dealloc(rB); _dealloc(vB);
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

void CurvatureFish::computeMidline(const Real t, const Real dt)
{
  const std::array<Real ,6> curvaturePoints = { (Real)0, (Real).15*length,
    (Real).4*length, (Real).65*length, (Real).9*length, length
  };
  const std::array<Real ,6> curvatureValues = {
      (Real)0.82014/length, (Real)1.46515/length, (Real)2.57136/length,
      (Real)3.75425/length, (Real)5.09147/length, (Real)5.70449/length
  };
  const std::array<Real,7> bendPoints = {(Real)-.5, (Real)-.25, (Real)0,
    (Real).25, (Real).5, (Real).75, (Real)1};

  #if 1 // ramp-up over Tperiod
  const std::array<Real,6> curvatureZeros = std::array<Real, 6>();
  curvatureScheduler.transition(0,0,Tperiod,curvatureZeros ,curvatureValues);
  #else // no rampup for debug
  curvatureScheduler.transition(t,0,Tperiod,curvatureValues,curvatureValues);
  #endif

  curvatureScheduler.gimmeValues(t,                curvaturePoints,Nm,rS,rC,vC);
  rlBendingScheduler.gimmeValues(t,periodPIDval,length, bendPoints,Nm,rS,rB,vB);

  // next term takes into account the derivative of periodPIDval in darg:
  const Real diffT = TperiodPID? 1 - (t-time0)*periodPIDdif/periodPIDval : 1;
  // time derivative of arg:
  const Real darg = 2*M_PI/periodPIDval * diffT;
  const Real arg0 = 2*M_PI*((t-time0)/periodPIDval +timeshift) +M_PI*phaseShift;

  #pragma omp parallel for schedule(static)
  for(int i=0; i<Nm; ++i) {
    const Real arg = arg0 - 2*M_PI*rS[i]/length/waveLength;
    rK[i] = amplitudeFactor* rC[i]*(std::sin(arg)     + rB[i] +curv_PID_fac);
    vK[i] = amplitudeFactor*(vC[i]*(std::sin(arg)     + rB[i] +curv_PID_fac)
                            +rC[i]*(std::cos(arg)*darg+ vB[i] +curv_PID_dif));
    assert(not std::isnan(rK[i]));
    assert(not std::isinf(rK[i]));
    assert(not std::isnan(vK[i]));
    assert(not std::isinf(vK[i]));
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
  CurvatureFish* const cFish = dynamic_cast<CurvatureFish*>( myFish );
  if(cFish == nullptr) { printf("Someone touched my fish\n"); abort(); }
  cFish->resetAll();
  Fish::resetAll();
}

StefanFish::StefanFish(SimulationData&s, ArgumentParser&p, double C[2]):
 Fish(s,p,C), bCorrectTrajectory(p("-pid").asInt(0)),
 bCorrectPosition(p("-pidpos").asInt(0))
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
  const double DT = sim.dt/Tperiod, time = sim.time;
  // Control pos diffs
  const double   xDiff = (centerOfMass[0] - origC[0])/length;
  const double   yDiff = (centerOfMass[1] - origC[1])/length;
  const double angDiff =  orientation     - origAng;
  const double relU = (u + sim.uinfx) / length;
  const double relV = (v + sim.uinfy) / length;
  const double angVel = omega, lastAngVel = cFish->lastAvel;
  // compute ang vel at t - 1/2 dt such that we have a better derivative:
  const double aVelMidP = (angVel + lastAngVel)*Tperiod/2;
  const double aVelDiff = (angVel - lastAngVel)*Tperiod/sim.dt;
  cFish->lastAvel = angVel; // store for next time

  // derivatives of following 2 exponential averages:
  const double velDAavg = (angDiff-cFish->avgDangle)/Tperiod + DT * angVel;
  const double velDYavg = (  yDiff-cFish->avgDeltaY)/Tperiod + DT * relV;
  const double velAVavg = 10*((aVelMidP-cFish->avgAngVel)/Tperiod +DT*aVelDiff);
  // exponential averages
  cFish->avgDangle = (1.0 -DT) * cFish->avgDangle +    DT * angDiff;
  cFish->avgDeltaY = (1.0 -DT) * cFish->avgDeltaY +    DT *   yDiff;
  // faster average:
  cFish->avgAngVel = (1-10*DT) * cFish->avgAngVel + 10*DT *aVelMidP;
  const double avgDangle = cFish->avgDangle, avgDeltaY = cFish->avgDeltaY;

  // integral (averaged) and proportional absolute DY and their derivative
  const double absPy = std::fabs(yDiff), absIy = std::fabs(avgDeltaY);
  const double velAbsPy =     yDiff>0 ? relV     : -relV;
  const double velAbsIy = avgDeltaY>0 ? velDYavg : -velDYavg;

  if (bCorrectPosition || bCorrectTrajectory)
    assert(origAng<2e-16 && "TODO: rotate pos and vel to fish POV to enable \
                             PID to work even for non-zero angles");

  if (bCorrectPosition && sim.dt>0)
  {
    //If angle is positive: positive curvature only if Dy<0 (must go up)
    //If angle is negative: negative curvature only if Dy>0 (must go down)
    const double IangPdy = (avgDangle *     yDiff < 0)? avgDangle * absPy : 0;
    const double PangIdy = (angDiff   * avgDeltaY < 0)? angDiff   * absIy : 0;
    const double IangIdy = (avgDangle * avgDeltaY < 0)? avgDangle * absIy : 0;

    // derivatives multiplied by 0 when term is inactive later:
    const double velIangPdy = velAbsPy * avgDangle + absPy * velDAavg;
    const double velPangIdy = velAbsIy * angDiff   + absIy * angVel;
    const double velIangIdy = velAbsIy * avgDangle + absIy * velDAavg;

    //zero also the derivatives when appropriate
    const double coefIangPdy = avgDangle *     yDiff < 0 ? 1 : 0;
    const double coefPangIdy = angDiff   * avgDeltaY < 0 ? 1 : 0;
    const double coefIangIdy = avgDangle * avgDeltaY < 0 ? 1 : 0;

    const double valIangPdy = coefIangPdy *    IangPdy;
    const double difIangPdy = coefIangPdy * velIangPdy;
    const double valPangIdy = coefPangIdy *    PangIdy;
    const double difPangIdy = coefPangIdy * velPangIdy;
    const double valIangIdy = coefIangIdy *    IangIdy;
    const double difIangIdy = coefIangIdy * velIangIdy;
    const double periodFac = 1.0 - xDiff;
    const double periodVel =     - relU;

    if(not sim.muteAll) {
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
    cFish->correctTrajectory(totalTerm, totalDiff, sim.time, sim.dt);
    cFish->correctTailPeriod(periodFac, periodVel, sim.time, sim.dt);
  }
  // if absIy<EPS then we have just one fish that the simulation box follows
  // therefore we control the average angle but not the Y disp (which is 0)
  else if (bCorrectTrajectory && sim.dt>0)
  {
    const double avgAngVel = cFish->avgAngVel, absAngVel = std::fabs(avgAngVel);
    const double absAvelDiff = avgAngVel>0? velAVavg : -velAVavg;
    const Real coefInst = angDiff*avgAngVel>0 ? 0.01 : 1, coefAvg = 0.1;
    const Real termInst = angDiff*absAngVel;
    const Real diffInst = angDiff*absAvelDiff + angVel*absAngVel;
    const double totalTerm = coefInst*termInst + coefAvg*avgDangle;
    const double totalDiff = coefInst*diffInst + coefAvg*velDAavg;

    if(not sim.muteAll) {
      std::ofstream filePID;
      std::stringstream ssF;
      ssF<<sim.path2file<<"/PID_"<<obstacleID<<".dat";
      filePID.open(ssF.str().c_str(), std::ios::app);
      filePID<<time<<" "<<coefInst*termInst<<" "<<coefInst*diffInst
                   <<" "<<coefAvg*avgDangle<<" "<<coefAvg*velDAavg<<"\n";
    }
    cFish->correctTrajectory(totalTerm, totalDiff, sim.time, sim.dt);
  }


  Fish::create(vInfo);
}

void StefanFish::act(const Real t_rlAction, const std::vector<double>& a) const
{
  CurvatureFish* const cFish = dynamic_cast<CurvatureFish*>( myFish );
  cFish->execute(sim.time, t_rlAction, a);
}

double StefanFish::getLearnTPeriod() const
{
  const CurvatureFish* const cFish = dynamic_cast<CurvatureFish*>( myFish );
  return cFish->periodPIDval;
}

double StefanFish::getPhase(const double t) const
{
  const CurvatureFish* const cFish = dynamic_cast<CurvatureFish*>( myFish );
  const double T0 = cFish->time0;
  const double Ts = cFish->timeshift;
  const double Tp = cFish->periodPIDval;
  const double arg  = 2*M_PI*((t-T0)/Tp +Ts) + M_PI*phaseShift;
  const double phase = std::fmod(arg, 2*M_PI);
  return (phase<0) ? 2*M_PI + phase : phase;
}

std::vector<double> StefanFish::state(Shape*const p) const
{
  const CurvatureFish* const cFish = dynamic_cast<CurvatureFish*>( myFish );
  std::vector<double> S(10,0);
  S[0] = ( center[0] - p->center[0] )/ length;
  S[1] = ( center[1] - p->center[1] )/ length;
  S[2] = getOrientation();
  S[3] = getPhase( sim.time );
  S[4] = getU() * Tperiod / length;
  S[5] = getV() * Tperiod / length;
  S[6] = getW() * Tperiod;
  S[7] = cFish->lastTact;
  S[8] = cFish->lastCurv;
  S[9] = cFish->oldrCurv;

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

      const auto &DU = myFish->upperSkin, &DL = myFish->lowerSkin;
      // first point of the two skins is the same
      // normal should be almost the same: take the mean
      const Real skinX=DU.xSurf[0], normX=(DU.normXSurf[0]+DL.normXSurf[0])/2;
      const Real skinY=DU.ySurf[0], normY=(DU.normYSurf[0]+DL.normYSurf[0])/2;
      const Real sensX = skinX + 2*h * normX, sensY = skinY + 2*h * normY;
      if( holdsPoint(velInfo[i], sensX, sensY) ) {

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
        const Real uSkin = u - omega*(skinY-centerOfMass[1]) + udef[iy][ix][0];
        const Real vSkin = v + omega*(skinX-centerOfMass[0]) + udef[iy][ix][1];
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
       if( holdsPoint(velInfo[i], sensX, sensY) ) {

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
        const Real uSkin = u - omega*(skinY-centerOfMass[1]) + udef[iy][ix][0];
        const Real vSkin = v + omega*(skinX-centerOfMass[0]) + udef[iy][ix][1];
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
