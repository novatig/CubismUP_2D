//
//  IF2D_ObstacleLibrary.h
//  IF2D_ROCKS
//
//  Created by Wim van Rees on 04/10/14.
//
//
#pragma once

#include "common.h"
#include "Definitions.h"
#include <map>
#include <limits>
#include <vector>
#include <array>
#include <fstream>

struct IF2D_Frenet2D
{
  static void solve( const unsigned Nm, const Real*const rS,
    const Real*const curv, const Real*const curv_dt,
    Real*const rX, Real*const rY, Real*const vX, Real*const vY,
    Real*const norX, Real*const norY, Real*const vNorX, Real*const vNorY )
  {
    // initial conditions
    rX[0] = 0.0;
    rY[0] = 0.0;
    norX[0] = 0.0;
    norY[0] = 1.0;
    Real ksiX = 1.0;
    Real ksiY = 0.0;
    // velocity variables
    vX[0] = 0.0;
    vY[0] = 0.0;
    vNorX[0] = 0.0;
    vNorY[0] = 0.0;
    Real vKsiX = 0.0;
    Real vKsiY = 0.0;

    for(unsigned i=1; i<Nm; i++) {
      // compute derivatives positions
      const Real dksiX = curv[i-1]*norX[i-1];
      const Real dksiY = curv[i-1]*norY[i-1];
      const Real dnuX = -curv[i-1]*ksiX;
      const Real dnuY = -curv[i-1]*ksiY;
      // compute derivatives velocity
      const Real dvKsiX = curv_dt[i-1]*norX[i-1] + curv[i-1]*vNorX[i-1];
      const Real dvKsiY = curv_dt[i-1]*norY[i-1] + curv[i-1]*vNorY[i-1];
      const Real dvNuX = -curv_dt[i-1]*ksiX - curv[i-1]*vKsiX;
      const Real dvNuY = -curv_dt[i-1]*ksiY - curv[i-1]*vKsiY;
      // compute current ds
      const Real ds = rS[i] - rS[i-1];
      // update
      rX[i] = rX[i-1] + ds*ksiX;
      rY[i] = rY[i-1] + ds*ksiY;
      norX[i] = norX[i-1] + ds*dnuX;
      norY[i] = norY[i-1] + ds*dnuY;
      ksiX += ds * dksiX;
      ksiY += ds * dksiY;
      // update velocities
      vX[i] = vX[i-1] + ds*vKsiX;
      vY[i] = vY[i-1] + ds*vKsiY;
      vNorX[i] = vNorX[i-1] + ds*dvNuX;
      vNorY[i] = vNorY[i-1] + ds*dvNuY;
      vKsiX += ds * dvKsiX;
      vKsiY += ds * dvKsiY;
      // normalize unit vectors
      const Real d1 = ksiX*ksiX + ksiY*ksiY;
      const Real d2 = norX[i]*norX[i] + norY[i]*norY[i];
      if(d1>std::numeric_limits<Real>::epsilon()) {
        const Real normfac = 1/std::sqrt(d1);
        ksiX*=normfac;
        ksiY*=normfac;
      }
      if(d2>std::numeric_limits<Real>::epsilon()) {
        const Real normfac = 1/std::sqrt(d2);
        norX[i]*=normfac;
        norY[i]*=normfac;
      }
    }
  }
};

class IF2D_Interpolation1D
{
 public:

  static void naturalCubicSpline(const Real*x, const Real*y,
    const unsigned n, const Real*xx, Real*yy, const unsigned nn) {
      return naturalCubicSpline(x,y,n,xx,yy,nn,0);
  }

  static void naturalCubicSpline(const Real*x, const Real*y, const unsigned n,
    const Real*xx, Real*yy, const unsigned nn, const Real offset) {
    Real y2[n];
    Real u[n-1];
    Real p, qn, sig, un, h, b, a;

    y2[0] = 0;
    u[0] = 0;
    for(unsigned i=1; i<n-1; i++) {
      sig = (x[i]-x[i-1])/(x[i+1]-x[i-1]);
      p = sig*y2[i-1] +2;
      y2[i] = (sig-1)/p;
      u[i] = (y[i+1]-y[i])/(x[i+1]-x[i])-(y[i]-y[i-1])/(x[i]-x[i-1]);
      u[i] = (6*u[i]/(x[i+1]-x[i-1])-sig*u[i-1])/p;
    }

    qn = 0;
    un = 0;
    y2[n-1] = (un-qn*u[n-2])/(qn*y2[n-2] +1);

    for(unsigned k=n-2; k>0; k--) y2[k] = y2[k]*y2[k+1] +u[k];

    for(unsigned j=0; j<nn; j++) {
      unsigned int klo = 0;
      unsigned int khi = n-1;
      unsigned int k = 0;
      while(khi-klo>1) {
        k=(khi+klo)>>1;
        if( x[k]>(xx[j]+offset)) khi=k;
        else                     klo=k;
      }

      h = x[khi] - x[klo];
      if(h<=0.0) {
        std::cout << "Interpolation points must be distinct!\n"; abort();
      }
      a = (x[khi]-(xx[j]+offset))/h;
      b = ((xx[j]+offset)-x[klo])/h;
      yy[j] = a*y[klo]+b*y[khi]+((a*a*a-a)*y2[klo]+(b*b*b-b)*y2[khi])*(h*h)/6;
    }
  }

  static void cubicInterpolation(const Real x0, const Real x1, const Real x,
    const Real y0,const Real y1,const Real dy0,const Real dy1, Real&y, Real&dy)
  {
    const Real xrel = (x-x0);
    const Real deltax = (x1-x0);

    const Real a = (dy0+dy1)/(deltax*deltax) - 2*(y1-y0)/(deltax*deltax*deltax);
    const Real b = (-2*dy0-dy1)/deltax + 3*(y1-y0)/(deltax*deltax);
    const Real c = dy0;
    const Real d = y0;

    y = a*xrel*xrel*xrel + b*xrel*xrel + c*xrel + d;
    dy = 3*a*xrel*xrel + 2*b*xrel + c;
  }

  static void cubicInterpolation(const Real x0, const Real x1, const Real x,
    const Real y0, const Real y1, Real & y, Real & dy) {
    return cubicInterpolation(x0,x1,x,y0,y1,0,0,y,dy); // 0 slope at end points
  }
};

namespace Schedulers
{
template<int Npoints>
struct ParameterScheduler
{
  std::array<Real, Npoints>  parameters_t0; // parameters at t0
  std::array<Real, Npoints>  parameters_t1; // parameters at t1
  std::array<Real, Npoints> dparameters_t0; // derivative at t0
  Real t0, t1; // t0 and t1

  void save(std::string filename) {
    std::ofstream savestream;
    savestream.setf(std::ios::scientific);
    savestream.precision(std::numeric_limits<Real>::digits10 + 1);
    savestream.open(filename+".txt");

    savestream << t0 << "\t" << t1 << std::endl;
    for(int i=0;i<Npoints;++i)
      savestream << parameters_t0[i]  << "\t"
                 << parameters_t1[i]  << "\t"
                 << dparameters_t0[i] << std::endl;
    savestream.close();
  }

  void restart(std::string filename) {
    std::ifstream restartstream;
    restartstream.open(filename+".txt");
    restartstream >> t0 >> t1;
    for(int i=0;i<Npoints;++i)
    restartstream >> parameters_t0[i] >> parameters_t1[i] >> dparameters_t0[i];
    restartstream.close();
  }
  virtual void resetAll() {
     parameters_t0 = std::array<Real, Npoints>();
     parameters_t1 = std::array<Real, Npoints>();
    dparameters_t0 = std::array<Real, Npoints>();
    t0 = -1;
    t1 =  0;
  }

  ParameterScheduler()
  {
    t0=-1; t1=0;
     parameters_t0 = std::array<Real, Npoints>();
     parameters_t1 = std::array<Real, Npoints>();
    dparameters_t0 = std::array<Real, Npoints>();
  }

  void transition(const Real t, const Real tstart, const Real tend,
      const std::array<Real, Npoints> parameters_tend,
      const bool UseCurrentDerivative = false)
  {
    if(t<tstart or t>tend) return; // this transition is out of scope
    //if(tstart<t0) return; // this transition is not relevant: we are doing a next one already

    // we transition from whatever state we are in to a new state
    // the start point is where we are now: lets find out
    std::array<Real, Npoints> parameters;
    std::array<Real, Npoints> dparameters;
    gimmeValues(tstart,parameters,dparameters);

    // fill my members
    t0 = tstart;
    t1 = tend;
    parameters_t0 = parameters;
    parameters_t1 = parameters_tend;
    dparameters_t0 = UseCurrentDerivative ? dparameters : std::array<Real, Npoints>();
  }

  void transition(const Real t, const Real tstart, const Real tend,
      const std::array<Real, Npoints> parameters_tstart,
      const std::array<Real, Npoints> parameters_tend)
  {
    if(t<tstart or t>tend) return; // this transition is out of scope
    if(tstart<t0) return; // this transition is not relevant: we are doing a next one already

    // fill my members
    t0 = tstart;
    t1 = tend;
    parameters_t0 = parameters_tstart;
    parameters_t1 = parameters_tend;
  }

  void gimmeValues(const Real t, std::array<Real, Npoints>& parameters, std::array<Real, Npoints>& dparameters)
  {
    // look at the different cases
    if(t<t0 or t0<0) { // no transition, we are in state 0
      parameters = parameters_t0;
      dparameters = std::array<Real, Npoints>();
    } else if(t>t1) { // no transition, we are in state 1
      parameters = parameters_t1;
      dparameters = std::array<Real, Npoints>();
    } else { // we are within transition: interpolate
      for(int i=0;i<Npoints;++i)
        IF2D_Interpolation1D::cubicInterpolation(t0,t1,t,parameters_t0[i],parameters_t1[i],dparameters_t0[i],0.0,parameters[i],dparameters[i]);
    }
  }

  void gimmeValues(const Real t, std::array<Real, Npoints>& parameters)
  {
    std::array<Real, Npoints> dparameters_whocares; // no derivative info
    return gimmeValues(t,parameters,dparameters_whocares);
  }
};

struct ParameterSchedulerScalar : ParameterScheduler<1>
{
  void transition(const Real t, const Real tstart, const Real tend,
    const Real parameter_tend, const bool keepSlope = false) {
    const std::array<Real, 1> myParameter = {parameter_tend};
    return
      ParameterScheduler<1>::transition(t,tstart,tend,myParameter,keepSlope);
  }

  void gimmeValues(const Real t, Real & parameter, Real & dparameter)
  {
    std::array<Real, 1> myParameter, mydParameter;
    ParameterScheduler<1>::gimmeValues(t, myParameter, mydParameter);
    parameter = myParameter[0];
    dparameter = mydParameter[0];
  }

  void gimmeValues(const Real t, Real & parameter)
  {
    std::array<Real, 1> myParameter;
    ParameterScheduler<1>::gimmeValues(t, myParameter);
    parameter = myParameter[0];
  }
};

template<int Npoints>
struct ParameterSchedulerVector : ParameterScheduler<Npoints>
{
  void gimmeValues(const Real t, const std::array<Real, Npoints>& positions,
    const int Nfine, const Real*const positions_fine,
    Real*const parameters_fine, Real * const dparameters_fine) {
    // we interpolate in space the start and end point
    Real* parameters_t0_fine  = new Real[Nfine];
    Real* parameters_t1_fine  = new Real[Nfine];
    Real* dparameters_t0_fine = new Real[Nfine];

    IF2D_Interpolation1D::naturalCubicSpline(positions.data(),
      this->parameters_t0.data(), Npoints, positions_fine, parameters_t0_fine,
      Nfine);
    IF2D_Interpolation1D::naturalCubicSpline(positions.data(),
      this->parameters_t1.data(), Npoints, positions_fine, parameters_t1_fine,
      Nfine);
    IF2D_Interpolation1D::naturalCubicSpline(positions.data(),
      this->dparameters_t0.data(),Npoints, positions_fine, dparameters_t0_fine,
      Nfine);

    // look at the different cases
    if(t<this->t0 or this->t0<0) {
      // no transition, we are in state 0
      for(int i=0;i<Nfine;++i) {
        parameters_fine[i] = parameters_t0_fine[i];
        dparameters_fine[i] = 0.0;
      }
    } else if(t>this->t1) {
      // no transition, we are in state 1
      for(int i=0;i<Nfine;++i) {
        parameters_fine[i] = parameters_t1_fine[i];
        dparameters_fine[i] = 0.0;
      }
    } else {
      // we are within transition: interpolate in time for each point of the fine discretization
      for(int i=0;i<Nfine;++i)
        IF2D_Interpolation1D::cubicInterpolation(this->t0, this->t1, t,
          parameters_t0_fine[i], parameters_t1_fine[i], dparameters_t0_fine[i],
          0, parameters_fine[i], dparameters_fine[i]);
    }
    delete [] parameters_t0_fine;
    delete [] parameters_t1_fine;
    delete [] dparameters_t0_fine;
  }

  void gimmeValues(const Real t, std::array<Real, Npoints>& parameters) {
    ParameterScheduler<Npoints>::gimmeValues(t, parameters);
  }

  void gimmeValues(const Real t, std::array<Real, Npoints> & parameters, std::array<Real, Npoints> & dparameters) {
    ParameterScheduler<Npoints>::gimmeValues(t, parameters, dparameters);
  }
};

template<int Npoints>
struct ParameterSchedulerLearnWave : ParameterScheduler<Npoints>
{
  template<typename T>
  void gimmeValues(const Real t, const Real Twave, const Real Length,
    const std::array<Real, Npoints> & positions, const int Nfine,
    const T*const positions_fine, T*const parameters_fine, Real*const dparameters_fine)
  {
    const Real _1oL = 1./Length;
    const Real _1oT = 1./Twave;
    // the fish goes through (as function of t and s) a wave function that describes the curvature
    for(int i=0;i<Nfine;++i) {
      const Real c = positions_fine[i]*_1oL - (t - this->t0)*_1oT; //traveling wave coord
      bool bCheck = true;

      if (c < positions[0]) { // Are you before latest wave node?
        IF2D_Interpolation1D::cubicInterpolation(
          c, positions[0], c,
          this->parameters_t0[0], this->parameters_t0[0],
          parameters_fine[i], dparameters_fine[i]);
        bCheck = false;
      }
      else if (c > positions[Npoints-1]) {// Are you after oldest wave node?
        IF2D_Interpolation1D::cubicInterpolation(
          positions[Npoints-1], c, c,
          this->parameters_t0[Npoints-1], this->parameters_t0[Npoints-1],
          parameters_fine[i], dparameters_fine[i]);
        bCheck = false;
      } else {
        for (int j=1; j<Npoints; ++j) { // Check at which point of the travelling wave we are
          if (( c >= positions[j-1] ) && ( c <= positions[j] )) {
            IF2D_Interpolation1D::cubicInterpolation(
              positions[j-1], positions[j], c,
              this->parameters_t0[j-1], this->parameters_t0[j],
              parameters_fine[i], dparameters_fine[i]);
            dparameters_fine[i] = -dparameters_fine[i]*_1oT; // df/dc * dc/dt
            bCheck = false;
          }
        }
      }
      if (bCheck) { std::cout << "Ciaone2!" << std::endl; abort(); }
    }
  }

  void Turn(const Real b, const Real t_turn) // each decision adds a node at the beginning of the wave (left, right, straight) and pops last node
  {
    this->t0 = t_turn;
    for(int i=Npoints-1; i>1; --i)
        this->parameters_t0[i] = this->parameters_t0[i-2];
    this->parameters_t0[1] = b;
    this->parameters_t0[0] = 0;
  }
};
}
