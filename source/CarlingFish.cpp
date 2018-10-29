//
//  IF2D_StefanFishOperator.cpp
//  IF2D_ROCKS
//
//  Created by Guido Novati on 01/07/15.
//
//

#include "CarlingFish.h"
#include "FishLibrary.h"
#include "FishUtilities.h"
#include <array>
#include <cmath>
#include <utility>
#include <time.h>
#include <random>

class AmplitudeFish : public FishData
{
 public:
  inline Real midlineLatPos(const Real s, const Real t) const {
    const Real arg = 2*M_PI*(s/length - t/Tperiod + phaseShift);
    return 4./33. *  (s + 0.03125*length)*std::sin(arg);
  }

  inline Real midlineLatVel(const Real s, const Real t) const {
      const Real arg = 2*M_PI*(s/length - t/Tperiod + phaseShift);
      return - 4./33. * (s + 0.03125*length) * (2*M_PI/Tperiod) * std::cos(arg);
  }

  inline Real rampFactorSine(const Real t, const Real T) const {
    return (t<T ? std::sin(0.5*M_PI*t/T) : 1.0);
  }

  inline Real rampFactorVelSine(const Real t, const Real T) const {
    return (t<T ? 0.5*M_PI/T * std::cos(0.5*M_PI*t/T) : 0.0);
  }

  AmplitudeFish(Real L, Real T, Real phi, Real _h, Real _A)
  : FishData(L, T, phi, _h, _A) { _computeWidth(); }

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

void AmplitudeFish::computeMidline(const Real t, const Real dt)
{
  const Real rampFac    = rampFactorSine(t, Tperiod);
  const Real rampFacVel = rampFactorVelSine(t, Tperiod);
  rX[0] = 0.0;
  rY[0] = rampFac * midlineLatPos(rS[0],t);
  vX[0] = 0.0; //rX[0] is constant
  vY[0] = rampFac*midlineLatVel(rS[0],t) + rampFacVel*midlineLatPos(rS[0],t);

  #pragma omp parallel for schedule(static)
  for(int i=1; i<Nm; ++i) {
    rY[i] = rampFac*midlineLatPos(rS[i],t);
    vY[i] = rampFac*midlineLatVel(rS[i],t) + rampFacVel*midlineLatPos(rS[i],t);
  }

  #pragma omp parallel for schedule(static)
  for(int i=1; i<Nm; ++i) {
    const Real dy = rY[i]-rY[i-1], ds = rS[i]-rS[i-1];
    const Real dx = std::sqrt(ds*ds-dy*dy);
    assert(dx>0);
    const Real dVy = vY[i]-vY[i-1];
    const Real dVx = - dy/dx * dVy; // ds^2 = dx^2+dy^2 -> ddx = -dy/dx*ddy

    rX[i] = dx;
    vX[i] = dVx;
    norX[ i-1] = -dy/ds;
    norY[ i-1] =  dx/ds;
    vNorX[i-1] = -dVy/ds;
    vNorY[i-1] =  dVx/ds;
  }

  for(int i=1; i<Nm; ++i) { rX[i] += rX[i-1]; vX[i] += vX[i-1]; }

  norX[Nm-1] = norX[Nm-2];
  norY[Nm-1] = norY[Nm-2];
  vNorX[Nm-1] = vNorX[Nm-2];
  vNorY[Nm-1] = vNorY[Nm-2];
}

CarlingFish::CarlingFish(SimulationData&s, ArgumentParser&p, double C[2])
  : Fish(s,p,C) {
  const Real ampFac = p("-amplitudeFactor").asDouble(1.0);
  myFish = new AmplitudeFish(length, Tperiod, phaseShift, sim.getH(), ampFac);
  printf("AmplitudeFish %d %f %f %f\n",myFish->Nm, length, Tperiod, phaseShift);
}

double CarlingFish::getPhase(const double t) const {
  const double Tp = myFish->l_Tp;
  const double T0 = myFish->time0;
  const double Ts = myFish->timeshift;
  const double arg  = 2*M_PI*((t-T0)/Tp +Ts) + M_PI*phaseShift;
  const double phase = std::fmod(arg, 2*M_PI);
  return (phase<0) ? 2*M_PI + phase : phase;
}

void CarlingFish::create(const vector<BlockInfo>& vInfo) {
  Fish::create(vInfo);
}

void CarlingFish::resetAll() {
  Fish::resetAll();
}
