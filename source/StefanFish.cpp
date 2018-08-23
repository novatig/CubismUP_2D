//
//  IF2D_StefanFishOperator.cpp
//  IF2D_ROCKS
//
//  Created by Guido Novati on 01/07/15.
//
//

#include "StefanFish.h"
#include "FishLibrary.h"
#include "FishUtilities.h"
#include <array>
#include <cmath>
#include <utility>
#include <time.h>
#include <random>

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
  Real controlFac = -1, valPID = 0;
  Real controlVel  = 0, velPID = 0;
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

  void _correctTrajectory(const Real dtheta, const Real time, Real dt);

  void _correctAmplitude(Real dAmp, Real vAmp, const Real time, const Real dt);

  void execute(const Real time, const Real l_tnext, const vector<double>& input);

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
  controlFac = -1; valPID = 0;
  controlVel  = 0; velPID = 0;
  curvScheduler.resetAll();
  baseScheduler.resetAll();
  adjustScheduler.resetAll();
  FishData::resetAll();
}

void CurvatureFish::_correctTrajectory(const Real dtheta, const Real t, Real dt)
{
  valPID = dtheta;
  dt = std::max(std::numeric_limits<Real>::epsilon(),dt);
  std::array<Real, 6> tmp_curv;
  tmp_curv.fill(dtheta);
  //adjustScheduler.transition(time,time,time+2*dt,tmp_curv, true);
  adjustScheduler.transition(t, t-10*dt, t+10*dt, tmp_curv, true);
}

void CurvatureFish::_correctAmplitude(Real dAmp, Real vAmp, const Real time, const Real dt)
{
  assert(dAmp>0 && dAmp<2); //buhu
  if(dAmp<=0) { dAmp=0; vAmp=0; }
  controlFac = dAmp;
  controlVel = vAmp;
  //TODO actually should be cubic spline!
  //const Real rampUp = time<Tperiod ? time/Tperiod : 1;
  //const Real fac = dAmp*rampUp/length; //curvature is 1/length
  //const std::array<Real ,6> curvature_values = {
  // fac*.82014, fac*1.46515, fac*2.57136, fac*3.75425, fac*5.09147, fac*5.70449
  //};
  //curvScheduler.transition(time,time,time+2*dt, curvature_values, true);
  //curvScheduler.transition(time, time-dt, time+dt, curvature_values);
}

void CurvatureFish::execute(const Real t,const Real lTact,const vector<double>&a)
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
  const Real _1oL = 1./length, _1oT = 1./l_Tp;
  const std::array<Real ,6> curvature_points = { (Real)0, (Real).15*length,
    (Real).4*length, (Real).65*length, (Real).9*length, length
  };
  const std::array<Real,7> baseline_points = {(Real)-.5, (Real)-.25, (Real)0,
    (Real).25, (Real).5, (Real).75, (Real)1};
  const std::array<Real ,6> curvature_values = {
      (Real)0.82014*_1oL, (Real)1.46515*_1oL, (Real)2.57136*_1oL,
      (Real)3.75425*_1oL, (Real)5.09147*_1oL, (Real)5.70449*_1oL
  };
  //const std::array<Real ,6> curvature_values = std::array<Real, 6>();
  const std::array<Real,6> curvature_zeros = std::array<Real, 6>();
  curvScheduler.transition(time,0,Tperiod,curvature_zeros,curvature_values);
  //curvScheduler.transition(time,0,Tperiod,curvature_values,curvature_values);

  // query the schedulers for current values
   curvScheduler.gimmeValues( time,            curvature_points, Nm,rS, rC,vC);
   baseScheduler.gimmeValues( time,l_Tp,length,baseline_points,  Nm,rS, rB,vB);
  adjustScheduler.gimmeValues(time,            curvature_points, Nm,rS, rA,vA);
  if(controlFac>0) {
    const Real _vA = velPID, _rA = valPID;
    #pragma omp parallel for schedule(static)
    for(int i=0; i<Nm; i++) {
      const Real darg = 2*M_PI* _1oT;
      const Real arg  = 2*M_PI*(_1oT*(time-time0) +timeshift
                                -rS[i]*_1oL/waveLength) + M_PI*phaseShift;
      rK[i] = amplitudeFactor* rC[i]*(std::sin(arg)     +rB[i]+_rA)*controlFac;
      vK[i] = amplitudeFactor*(vC[i]*(std::sin(arg)     +rB[i]+_rA)*controlFac
                              +rC[i]*(std::cos(arg)*darg+vB[i]+_vA)*controlFac
                              +rC[i]*(std::sin(arg)     +rB[i]+_rA)*controlVel);
    }
  } else {
    #pragma omp parallel for schedule(static)
    for(int i=0; i<Nm; i++) {
      const Real darg = 2*M_PI* _1oT;
      const Real arg  = 2*M_PI*(_1oT*(time-time0) +timeshift
                                -rS[i]*_1oL/waveLength) + M_PI*phaseShift;
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

StefanFish::StefanFish(SimulationData&s,ArgumentParser&p,Real C[2]):Fish(s,p,C),
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

static inline Real sgn(const Real val) { return (0 < val) - (val < 0); }
void StefanFish::create(const vector<BlockInfo>& vInfo)
{
  CurvatureFish* const cFish = dynamic_cast<CurvatureFish*>( myFish );
  if(cFish == nullptr) { printf("Someone touched my fish\n"); abort(); }

  if (bCorrectTrajectory) {
    assert(followX < 0 && followY < 0);
    const Real AngDiff  = std::atan2(-v, -u);
    adjTh = (Tperiod-sim.dt) * adjTh + sim.dt * AngDiff;
    const Real INST = (AngDiff*omega>0) ? AngDiff*std::fabs(omega) : 0;
    const Real PID  = 0.1*adjTh + 0.1*INST;
    if(not sim.muteAll) {
      ofstream filePID;
      stringstream ssF; ssF<<sim.path2file<<"/PID_"<<obstacleID<<".dat";
      filePID.open(ssF.str().c_str(), ios::app);
      filePID<<adjTh<<" "<<INST<<endl;
      filePID.close();
    }
    cFish->_correctTrajectory(PID, sim.time, sim.dt);
  }

  Fish::create(vInfo);
}

void StefanFish::act(const Real lTact, const vector<double>& a) const {
  CurvatureFish* const cFish = dynamic_cast<CurvatureFish*>( myFish );
  oldrCurv = lastCurv;
  lastCurv = a[0];
  if(a.size()>0) lastTact = a[1];
  cFish->execute(sim.time, lTact, a);
}

double StefanFish::getLearnTPeriod() const {
  return myFish->l_Tp;
}
double StefanFish::getPhase(const double t) const {
  const double Tp = myFish->l_Tp;
  const double T0 = myFish->time0;
  const double Ts = myFish->timeshift;
  const double arg  = 2*M_PI*((t-T0)/Tp +Ts) + M_PI*phaseShift;
  const double phase = std::fmod(arg, 2*M_PI);
  return (phase<0) ? 2*M_PI + phase : phase;
}
