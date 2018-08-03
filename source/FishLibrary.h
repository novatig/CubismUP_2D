//
//  IF2D_ObstacleLibrary.h
//  IF2D_ROCKS
//
//  Created by Wim van Rees on 04/10/14.
//
//
#pragma once

#include "ObstacleBlock.h"
#include "GenericOperator.h"
#include <map>
#include <limits>
#include <vector>
#include <array>
#include <fstream>

/*
struct FishSkin
{
 public:
  const int Npoints;
  Real * const xSurf;
  Real * const ySurf;
  Real * const normXSurf;
  Real * const normYSurf;
  Real * const midX;
  Real * const midY;

  FishSkin(const int N): Npoints(N), xSurf(_alloc(N)), ySurf(_alloc(N)),
    normXSurf(_alloc(N-1)), normYSurf(_alloc(N-1)), midX(_alloc(N-1)),
    midY(_alloc(N-1)) { }

  virtual ~FishSkin() {
      _dealloc(xSurf);
      _dealloc(ySurf);
      _dealloc(normXSurf);
      _dealloc(normYSurf);
      _dealloc(midX);
      _dealloc(midY);
  }
};
*/

struct FishData
{
 public:
  const Real length, Tperiod, phaseShift, h;
  const Real waveLength = 1;
  const Real amplitudeFactor;

  // Midline is discretized by more points in first fraction and last fraction:
  const double fracRefined = 0.1, fracMid = 1 - 2*fracRefined;
  const double dSmid_tgt = h / std::sqrt(2);
  const double dSrefine_tgt = 0.125 * h;

  const int Nmid = (int)std::ceil(length * fracMid / dSmid_tgt / 8) * 8;
  const Real dSmid = length * fracMid / Nmid;

  const int Nend = (int)std::ceil( // here we ceil to be safer
    fracRefined * length * 2 / (dSmid + dSrefine_tgt)  / 4) * 4;
  const double dSref = fracRefined * length * 2 / Nend - dSmid;

  const int Nm = Nmid + 2 * Nend + 1; // plus 1 because we contain 0 and L

  Real * const rS; // arclength discretization points
  Real * const rX; // coordinates of midline discretization points
  Real * const rY;
  Real * const vX; // midline discretization velocities
  Real * const vY;
  Real * const norX; // normal vector to the midline discretization points
  Real * const norY;
  Real * const vNorX;
  Real * const vNorY;
  Real * const width;
  Real oldTime = 0.0;
  // quantities needed to correctly control the speed of the midline maneuvers
  Real l_Tp = Tperiod, timeshift = 0, time0 = 0;

  Real linMom[2], area, J, angMom; // for diagnostics
  // start and end indices in the arrays where the fish starts and ends (to ignore the extensions when interpolating the shapes)
  //FishSkin * upperSkin, * lowerSkin;

 protected:
  Real Rmatrix2D[2][2];

  template<typename T>
  inline void _rotate2D(T &x, T &y) const {
    const T p[2] = {x, y};
    x = Rmatrix2D[0][0]*p[0] + Rmatrix2D[0][1]*p[1];
    y = Rmatrix2D[1][0]*p[0] + Rmatrix2D[1][1]*p[1];
  }
  template<typename T>
  inline void _translateAndRotate2D(const T pos[2], Real&x, Real&y) const {
    const Real p[2] = { x-pos[0], y-pos[1] };
    x = Rmatrix2D[0][0]*p[0] + Rmatrix2D[0][1]*p[1];
    y = Rmatrix2D[1][0]*p[0] + Rmatrix2D[1][1]*p[1];
  }
  void _prepareRotation2D(const Real angle) {
    Rmatrix2D[0][0] =  std::cos(angle);
    Rmatrix2D[0][1] = -std::sin(angle);
    Rmatrix2D[1][0] =  std::sin(angle);
    Rmatrix2D[1][1] =  std::cos(angle);
  }

  Real* _alloc(const int N) {
    return new Real[N];
  }
  template<typename T>
  void _dealloc(T * ptr) {
    if(ptr not_eq nullptr) { delete [] ptr; ptr=nullptr; }
  }
  inline Real _d_ds(const int idx, const Real*const vals, const int maxidx) const {
    if(idx==0) return (vals[idx+1]-vals[idx])/(rS[idx+1]-rS[idx]);
    else if(idx==maxidx-1) return (vals[idx]-vals[idx-1])/(rS[idx]-rS[idx-1]);
    else return ( (vals[idx+1]-vals[idx])/(rS[idx+1]-rS[idx]) +
                  (vals[idx]-vals[idx-1])/(rS[idx]-rS[idx-1]) )/2;
  }
  inline Real _integrationFac1(const int idx) const {
    return 2*width[idx];
  }
  inline Real _integrationFac2(const int idx) const {
    const Real dnorXi = _d_ds(idx, norX, Nm);
    const Real dnorYi = _d_ds(idx, norY, Nm);
    return 2*std::pow(width[idx],3)*(dnorXi*norY[idx] - dnorYi*norX[idx])/3;
  }
  inline Real _integrationFac3(const int idx) const {
    return 2*std::pow(width[idx],3)/3;
  }

  virtual void _computeMidlineNormals();

  virtual Real _width(const Real s, const Real L) = 0;

  void _computeWidth() {
    for(int i=0; i<Nm; ++i) width[i] = _width(rS[i], length);
  }

 public:
  FishData(Real L, Real Tp, Real phi, Real _h, const Real A=1);
  virtual ~FishData();

  Real integrateLinearMomentum(Real CoM[2], Real vCoM[2]);
  Real integrateAngularMomentum(Real & angVel);

  void changeToCoMFrameLinear(const Real CoM_internal[2], const Real vCoM_internal[2]);
  void changeToCoMFrameAngular(const Real theta_internal, const Real angvel_internal);

  //void computeSurface();
  //void surfaceToCOMFrame(const Real theta_internal, const Real CoM_internal[2]);
  //void surfaceToComputationalFrame(const Real theta_comp, const Real CoM_interpolated[2]);

  void writeMidline2File(const int step_id, std::string filename);

  virtual void computeMidline(const Real time, const Real dt) = 0;
};

struct AreaSegment
{
  const Real safe_distance;
  const std::pair<int, int> s_range;
  Real w[2], c[2];
  // should be normalized and >=0:
  Real normalI[2] = {(Real)1 , (Real)0};
  Real normalJ[2] = {(Real)0 , (Real)1};
  Real objBoxLabFr[2][2] = {{0,0}, {0,0}};
  Real objBoxObjFr[2][2] = {{0,0}, {0,0}};

  AreaSegment(std::pair<int,int> sr,const Real bb[2][2],const Real safe):
  safe_distance(safe), s_range(sr),
  w{ (bb[0][1]-bb[0][0])/2 + safe, (bb[1][1]-bb[1][0])/2 + safe },
  c{ (bb[0][1]+bb[0][0])/2,        (bb[1][1]+bb[1][0])/2 }
  { assert(w[0]>0); assert(w[1]>0); }

  void changeToComputationalFrame(const Real position[2], const Real angle);
  bool isIntersectingWithAABB(const Real start[2],const Real end[2]) const;
};

struct PutFishOnBlocks
{
  const FishData & cfish;
  const Real position[2];
  const Real angle;
  const Real Rmatrix2D[2][2] = {
      {std::cos(angle), -std::sin(angle)},
      {std::sin(angle),  std::cos(angle)}
  };
  static inline Real eulerDistSq2D(const Real a[2], const Real b[2]) {
    return std::pow(a[0]-b[0],2) +std::pow(a[1]-b[1],2);
  }
  void changeVelocityToComputationalFrame(Real x[2]) const {
    const Real p[2] = {x[0], x[1]};
    x[0]=Rmatrix2D[0][0]*p[0] + Rmatrix2D[0][1]*p[1]; // rotate (around CoM)
    x[1]=Rmatrix2D[1][0]*p[0] + Rmatrix2D[1][1]*p[1];
  }
  template<typename T>
  void changeToComputationalFrame(T x[2]) const {
    const T p[2] = {x[0], x[1]};
    x[0] = Rmatrix2D[0][0]*p[0] + Rmatrix2D[0][1]*p[1];
    x[1] = Rmatrix2D[1][0]*p[0] + Rmatrix2D[1][1]*p[1];
    x[0]+= position[0]; // translate
    x[1]+= position[1];
  }
  template<typename T>
  void changeFromComputationalFrame(T x[2]) const {
    const T p[2] = { x[0]-(T)position[0], x[1]-(T)position[1] };
    // rotate back around CoM
    x[0]=Rmatrix2D[0][0]*p[0] + Rmatrix2D[1][0]*p[1];
    x[1]=Rmatrix2D[0][1]*p[0] + Rmatrix2D[1][1]*p[1];
  }
  static constexpr Real EPS = std::numeric_limits<Real>::epsilon();

  PutFishOnBlocks(const FishData& cf, const Real pos[2],
    const Real ang): cfish(cf), position{pos[0],pos[1]}, angle(ang) { }

  void operator()(const BlockInfo& i, FluidBlock& b,
    ObstacleBlock* const o, const vector<AreaSegment*>& v) const;
  virtual void constructSurface(  const BlockInfo& i, FluidBlock& b,
    ObstacleBlock* const o, const vector<AreaSegment*>& v) const;
  virtual void constructInternl(  const BlockInfo& i, FluidBlock& b,
    ObstacleBlock* const o, const vector<AreaSegment*>& v) const;
  virtual void signedDistanceSqrt(const BlockInfo& i, FluidBlock& b,
    ObstacleBlock* const o, const vector<AreaSegment*>& v) const;
};

struct PutFishOnBlocks_Finalize : public GenericLabOperator
{
  const int stencil_start[3] = { -1, -1, 0};
  const int stencil_end[3]   = {  2,  2, 1};

  void operator()(Lab&l ,const BlockInfo&i, FluidBlock&b, ObstacleBlock*const o) const;
};
