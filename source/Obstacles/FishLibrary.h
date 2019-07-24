//
//  CubismUP_2D
//  Copyright (c) 2018 CSE-Lab, ETH Zurich, Switzerland.
//  Distributed under the terms of the MIT license.
//
//  Created by Guido Novati (novatig@ethz.ch).
//


#pragma once

#include "ObstacleBlock.h"
#include "../Operator.h"

struct FishSkin
{
  const size_t Npoints;
  Real * const xSurf;
  Real * const ySurf;
  Real * const normXSurf;
  Real * const normYSurf;
  Real * const midX;
  Real * const midY;
  FishSkin(const FishSkin& c) : Npoints(c.Npoints),
    xSurf(new Real[Npoints]), ySurf(new Real[Npoints])
    , normXSurf(new Real[Npoints-1]), normYSurf(new Real[Npoints-1])
    , midX(new Real[Npoints-1]), midY(new Real[Npoints-1])
    { }

  FishSkin(const size_t N): Npoints(N),
    xSurf(new Real[Npoints]), ySurf(new Real[Npoints])
    , normXSurf(new Real[Npoints-1]), normYSurf(new Real[Npoints-1])
    , midX(new Real[Npoints-1]), midY(new Real[Npoints-1])
    { }

  ~FishSkin() { delete [] xSurf; delete [] ySurf;
      delete [] normXSurf; delete [] normYSurf; delete [] midX; delete [] midY;
  }
};

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

  Real linMom[2], area, J, angMom; // for diagnostics
  // start and end indices in the arrays where the fish starts and ends (to ignore the extensions when interpolating the shapes)
  FishSkin upperSkin = FishSkin(Nm);
  FishSkin lowerSkin = FishSkin(Nm);
  virtual void resetAll();

 protected:

  template<typename T>
  inline void _rotate2D(const Real Rmatrix2D[2][2], T &x, T &y) const {
    const T p[2] = {x, y};
    x = Rmatrix2D[0][0]*p[0] + Rmatrix2D[0][1]*p[1];
    y = Rmatrix2D[1][0]*p[0] + Rmatrix2D[1][1]*p[1];
  }
  template<typename T>
  inline void _translateAndRotate2D(const T pos[2], const Real Rmatrix2D[2][2], Real&x, Real&y) const {
    const Real p[2] = { x-pos[0], y-pos[1] };
    x = Rmatrix2D[0][0]*p[0] + Rmatrix2D[0][1]*p[1];
    y = Rmatrix2D[1][0]*p[0] + Rmatrix2D[1][1]*p[1];
  }

  static Real* _alloc(const int N) {
    return new Real[N];
  }
  template<typename T>
  static void _dealloc(T * ptr) {
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

  virtual void _computeMidlineNormals() const;

  virtual Real _width(const Real s, const Real L) = 0;

  void _computeWidth() {
    for(int i=0; i<Nm; ++i) width[i] = _width(rS[i], length);
  }

 public:
  FishData(Real L, Real Tp, Real phi, Real _h, const Real A=1);
  virtual ~FishData();

  Real integrateLinearMomentum(double CoM[2], double vCoM[2]);
  Real integrateAngularMomentum(double & angVel);

  void changeToCoMFrameLinear(const double CoM_internal[2], const double vCoM_internal[2]) const;
  void changeToCoMFrameAngular(const Real theta_internal, const Real angvel_internal) const;

  void computeSurface() const;
  void surfaceToCOMFrame(const Real theta_internal,
                         const Real CoM_internal[2]) const;
  void surfaceToComputationalFrame(const Real theta_comp,
                                   const Real CoM_interpolated[2]) const;
  void computeSkinNormals(const Real theta_comp, const Real CoM_comp[3]) const;
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

  void changeToComputationalFrame(const double position[2], const Real angle);
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

  PutFishOnBlocks(const FishData& cf, const double pos[2],
    const Real ang): cfish(cf), position{(Real)pos[0],(Real)pos[1]}, angle(ang) { }

  void operator()(const cubism::BlockInfo& i, ScalarBlock& b,
    ObstacleBlock* const o, const std::vector<AreaSegment*>& v) const;
  virtual void constructSurface(  const cubism::BlockInfo& i, ScalarBlock& b,
    ObstacleBlock* const o, const std::vector<AreaSegment*>& v) const;
  virtual void constructInternl(  const cubism::BlockInfo& i, ScalarBlock& b,
    ObstacleBlock* const o, const std::vector<AreaSegment*>& v) const;
  virtual void signedDistanceSqrt(const cubism::BlockInfo& i, ScalarBlock& b,
    ObstacleBlock* const o, const std::vector<AreaSegment*>& v) const;
};
