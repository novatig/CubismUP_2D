#include "FishLibrary.h"
#include "FishUtilities.h"

FishData::FishData(Real L, Real Tp, Real phi, Real _h, const Real A):
 length(L), Tperiod(Tp), phaseShift(phi), h(_h), amplitudeFactor(A),
 rS(_alloc(Nm)),rX(_alloc(Nm)),rY(_alloc(Nm)),vX(_alloc(Nm)),vY(_alloc(Nm)),
 norX(_alloc(Nm)), norY(_alloc(Nm)), vNorX(_alloc(Nm)), vNorY(_alloc(Nm)),
 width(_alloc(Nm))//, upperSkin(new FishSkin(Nm)), lowerSkin(new FishSkin(Nm))
{
  // extension head
  assert(dSref > 0);
  rS[0] = 0;
  int k = 0;
  for(int i=0; i<Nend; ++i, k++)
    rS[k+1] = rS[k] + dSref +(dSmid-dSref) *         i /((double)Nend-1.);
  // interior points
  for(int i=0; i<Nmid; ++i, k++) rS[k+1] = rS[k] + dSmid;
  // extension tail
  for(int i=0; i<Nend; ++i, k++)
    rS[k+1] = rS[k] + dSref +(dSmid-dSref) * (Nend-i-1)/((double)Nend-1.);
  assert(k+1==Nm);
  //cout << "Discrepancy of midline length: " << std::fabs(rS[k]-L) << endl;
  rS[k] = std::min(rS[k], (Real)L);
  std::fill(rX, rX+Nm, 0);
  std::fill(rY, rY+Nm, 0);
  std::fill(vX, vX+Nm, 0);
  std::fill(vY, vY+Nm, 0);
}
FishData::~FishData() {
  _dealloc(rS); _dealloc(rX); _dealloc(rY); _dealloc(vX); _dealloc(vY);
  _dealloc(norX); _dealloc(norY); _dealloc(vNorX); _dealloc(vNorY);
  _dealloc(width);
  // if(upperSkin not_eq nullptr) { delete upperSkin; upperSkin=nullptr; }
  // if(lowerSkin not_eq nullptr) { delete lowerSkin; lowerSkin=nullptr; }
}
void FishData::resetAll() {
  l_Tp = Tperiod;
  timeshift = 0;
  time0 = 0;
}

void FishData::writeMidline2File(const int step_id, string filename) {
  char buf[500];
  sprintf(buf, "%s_midline_%07d.txt", filename.c_str(), step_id);
  FILE * f = fopen(buf, "w");
  fprintf(f, "s x y vX vY\n");
  for (int i=0; i<Nm; i++) {
    //dummy.changeToComputationalFrame(temp);
    //dummy.changeVelocityToComputationalFrame(udef);
    fprintf(f, "%g %g %g %g %g %g\n", rS[i],rX[i],rY[i],vX[i],vY[i],width[i]);
  }
}

void FishData::_computeMidlineNormals() {
  #pragma omp parallel for schedule(static)
  for(int i=0; i<Nm-1; i++) {
    const auto ds = rS[i+1]-rS[i];
    const auto tX = rX[i+1]-rX[i];
    const auto tY = rY[i+1]-rY[i];
    const auto tVX = vX[i+1]-vX[i];
    const auto tVY = vY[i+1]-vY[i];
    norX[i] = -tY/ds;
    norY[i] =  tX/ds;
    vNorX[i] = -tVY/ds;
    vNorY[i] =  tVX/ds;
  }
  norX[Nm-1] = norX[Nm-2];
  norY[Nm-1] = norY[Nm-2];
  vNorX[Nm-1] = vNorX[Nm-2];
  vNorY[Nm-1] = vNorY[Nm-2];
}

Real FishData::integrateLinearMomentum(double CoM[2], double vCoM[2]) {
  // already worked out the integrals for r, theta on paper
  // remaining integral done with composite trapezoidal rule
  // minimize rhs evaluations --> do first and last point separately
  double _area=0, _cmx=0, _cmy=0, _lmx=0, _lmy=0;
  #pragma omp parallel for schedule(static) reduction(+:_area,_cmx,_cmy,_lmx,_lmy)
  for(int i=0; i<Nm; ++i) {
    const double ds = (i==0) ? rS[1]-rS[0] :
        ((i==Nm-1) ? rS[Nm-1]-rS[Nm-2] :rS[i+1]-rS[i-1]);
    const double fac1 = _integrationFac1(i);
    const double fac2 = _integrationFac2(i);
    _area +=                        fac1 *ds/2;
    _cmx  += (rX[i]*fac1 +  norX[i]*fac2)*ds/2;
    _cmy  += (rY[i]*fac1 +  norY[i]*fac2)*ds/2;
    _lmx  += (vX[i]*fac1 + vNorX[i]*fac2)*ds/2;
    _lmy  += (vY[i]*fac1 + vNorY[i]*fac2)*ds/2;
  }
  area      = _area;
  CoM[0]    = _cmx;
  CoM[1]    = _cmy;
  linMom[0] = _lmx;
  linMom[1] = _lmy;
  assert(area> std::numeric_limits<Real>::epsilon());
  CoM[0] /= area;
  CoM[1] /= area;
  vCoM[0] = linMom[0]/area;
  vCoM[1] = linMom[1]/area;
  //printf("%f %f %f %f %f\n",CoM[0],CoM[1],vCoM[0],vCoM[1], vol);
  return area;
}
Real FishData::integrateAngularMomentum(double& angVel) {
  // assume we have already translated CoM and vCoM to nullify linear momentum
  // already worked out the integrals for r, theta on paper
  // remaining integral done with composite trapezoidal rule
  // minimize rhs evaluations --> do first and last point separately
  double _J = 0, _am = 0;
  #pragma omp parallel for reduction(+:_J,_am) schedule(static)
  for(int i=0; i<Nm; ++i) {
    const double ds =   (i==   0) ? rS[1]   -rS[0] :
                    ( (i==Nm-1) ? rS[Nm-1]-rS[Nm-2]
                                : rS[i+1] -rS[i-1] );
    const double fac1 = _integrationFac1(i);
    const double fac2 = _integrationFac2(i);
    const double fac3 = _integrationFac3(i);
    const double tmp_M = (rX[i]*vY[i] - rY[i]*vX[i])*fac1
      + (rX[i]*vNorY[i] -rY[i]*vNorX[i] +vY[i]*norX[i] -vX[i]*norY[i])*fac2
      + (norX[i]*vNorY[i] - norY[i]*vNorX[i])*fac3;

    const double tmp_J = (rX[i]*rX[i]   + rY[i]*rY[i]  )*fac1
                   + 2*(rX[i]*norX[i] + rY[i]*norY[i])*fac2 + fac3;

    _am += tmp_M*ds/2;
    _J  += tmp_J*ds/2;
  }
  J      = _J;
  angMom = _am;
  assert(J>std::numeric_limits<Real>::epsilon());
  angVel = angMom/J;
  return J;
}

void FishData::changeToCoMFrameLinear(const double CoMin[2],const double vCoMin[2]){
  #pragma omp parallel for schedule(static)
  for(int i=0;i<Nm;++i) {
   rX[i] -= CoMin[0]; rY[i] -= CoMin[1]; vX[i] -= vCoMin[0]; vY[i] -= vCoMin[1];
  }
}
void FishData::changeToCoMFrameAngular(const Real theta_internal, const Real angvel_internal) {
  _prepareRotation2D(theta_internal);
  #pragma omp parallel for schedule(static)
  for(int i=0;i<Nm;++i) {
    vX[i] += angvel_internal*rY[i];
    vY[i] -= angvel_internal*rX[i];
    _rotate2D(rX[i],rY[i]);
    _rotate2D(vX[i],vY[i]);
  }
  _computeMidlineNormals();
}

#if 0
void FishMidlineData::computeSurface() {
  const int Nskin = lowerSkin->Npoints;
  // Compute surface points by adding width to the midline points
  #pragma omp parallel for schedule(static)
  for(int i=0; i<Nskin; ++i) {
    Real norm[2] = {norX[i], norY[i]};
    Real const norm_mod1 = std::sqrt(norm[0]*norm[0] + norm[1]*norm[1]);
    norm[0] /= norm_mod1;
    norm[1] /= norm_mod1;
    assert(width[i] >= 0);
    lowerSkin->xSurf[i] = rX[i] - width[i]*norm[0];
    lowerSkin->ySurf[i] = rY[i] - width[i]*norm[1];
    upperSkin->xSurf[i] = rX[i] + width[i]*norm[0];
    upperSkin->ySurf[i] = rY[i] + width[i]*norm[1];
  }
}
void FishMidlineData::computeSkinNormals(const Real theta_comp, const Real CoM_comp[3]) {
  _prepareRotation2D(theta_comp);
  for(int i=0; i<Nm; ++i) {
    _rotate2D(rX[i], rY[i]);
    rX[i] += CoM_comp[0];
    rY[i] += CoM_comp[1];
  }

  const int Nskin = lowerSkin->Npoints;
  // Compute midpoints as they will be pressure targets
  #pragma omp parallel for
  for(int i=0; i<Nskin-1; ++i)
  {
    lowerSkin->midX[i] = (lowerSkin->xSurf[i] + lowerSkin->xSurf[i+1])/2.;
    upperSkin->midX[i] = (upperSkin->xSurf[i] + upperSkin->xSurf[i+1])/2.;
    lowerSkin->midY[i] = (lowerSkin->ySurf[i] + lowerSkin->ySurf[i+1])/2.;
    upperSkin->midY[i] = (upperSkin->ySurf[i] + upperSkin->ySurf[i+1])/2.;

    lowerSkin->normXSurf[i]=  (lowerSkin->ySurf[i+1]-lowerSkin->ySurf[i]);
    upperSkin->normXSurf[i]=  (upperSkin->ySurf[i+1]-upperSkin->ySurf[i]);
    lowerSkin->normYSurf[i]= -(lowerSkin->xSurf[i+1]-lowerSkin->xSurf[i]);
    upperSkin->normYSurf[i]= -(upperSkin->xSurf[i+1]-upperSkin->xSurf[i]);

    const Real normL = std::sqrt( std::pow(lowerSkin->normXSurf[i],2) +
                                  std::pow(lowerSkin->normYSurf[i],2) );
    const Real normU = std::sqrt( std::pow(upperSkin->normXSurf[i],2) +
                                  std::pow(upperSkin->normYSurf[i],2) );

    lowerSkin->normXSurf[i] /= normL;
    upperSkin->normXSurf[i] /= normU;
    lowerSkin->normYSurf[i] /= normL;
    upperSkin->normYSurf[i] /= normU;

    //if too close to the head or tail, consider a point further in, so that we are pointing out for sure
    const int ii = (i<8) ? 8 : ((i > Nskin-9) ? Nskin-9 : i);

    const Real dirL =
      lowerSkin->normXSurf[i] * (lowerSkin->midX[i]-rX[ii]) +
      lowerSkin->normYSurf[i] * (lowerSkin->midY[i]-rY[ii]);
    const Real dirU =
      upperSkin->normXSurf[i] * (upperSkin->midX[i]-rX[ii]) +
      upperSkin->normYSurf[i] * (upperSkin->midY[i]-rY[ii]);

    if(dirL < 0) {
        lowerSkin->normXSurf[i] *= -1.0;
        lowerSkin->normYSurf[i] *= -1.0;
    }
    if(dirU < 0) {
        upperSkin->normXSurf[i] *= -1.0;
        upperSkin->normYSurf[i] *= -1.0;
    }
  }
}
void FishMidlineData::surfaceToCOMFrame(const Real theta_internal, const Real CoM_internal[2]) {
  _prepareRotation2D(theta_internal);
  // Surface points rotation and translation

  #pragma omp parallel for
  for(int i=0; i<upperSkin->Npoints; ++i)
  //for(int i=0; i<upperSkin->Npoints-1; ++i)
  {
    upperSkin->xSurf[i] -= CoM_internal[0];
    upperSkin->ySurf[i] -= CoM_internal[1];
    _rotate2D(upperSkin->xSurf[i], upperSkin->ySurf[i]);
    lowerSkin->xSurf[i] -= CoM_internal[0];
    lowerSkin->ySurf[i] -= CoM_internal[1];
    _rotate2D(lowerSkin->xSurf[i], lowerSkin->ySurf[i]);
  }
}
void FishMidlineData::surfaceToComputationalFrame(const Real theta_comp, const Real CoM_interpolated[2]) {
  _prepareRotation2D(theta_comp);

  #pragma omp parallel for
  for(int i=0; i<upperSkin->Npoints; ++i)
  {
    _rotate2D(upperSkin->xSurf[i], upperSkin->ySurf[i]);
    upperSkin->xSurf[i] += CoM_interpolated[0];
    upperSkin->ySurf[i] += CoM_interpolated[1];
    _rotate2D(lowerSkin->xSurf[i], lowerSkin->ySurf[i]);
    lowerSkin->xSurf[i] += CoM_interpolated[0];
    lowerSkin->ySurf[i] += CoM_interpolated[1];
  }
}
#endif

void AreaSegment::changeToComputationalFrame(const double pos[2],const Real angle)
{
  // we are in CoM frame and change to comp frame --> first rotate around CoM (which is at (0,0) in CoM frame), then update center
  const Real Rmatrix2D[2][2] = {
      {std::cos(angle), -std::sin(angle)},
      {std::sin(angle),  std::cos(angle)}
  };
  const Real p[2] = {c[0],c[1]};

  const Real nx[2] = {normalI[0],normalI[1]};
  const Real ny[2] = {normalJ[0],normalJ[1]};

  for(int i=0;i<2;++i) {
      c[i] = Rmatrix2D[i][0]*p[0] + Rmatrix2D[i][1]*p[1];

      normalI[i] = Rmatrix2D[i][0]*nx[0] + Rmatrix2D[i][1]*nx[1];
      normalJ[i] = Rmatrix2D[i][0]*ny[0] + Rmatrix2D[i][1]*ny[1];
  }

  c[0] += pos[0];
  c[1] += pos[1];

  const Real magI = std::sqrt(normalI[0]*normalI[0]+normalI[1]*normalI[1]);
  const Real magJ = std::sqrt(normalJ[0]*normalJ[0]+normalJ[1]*normalJ[1]);
  assert(magI > std::numeric_limits<Real>::epsilon());
  assert(magJ > std::numeric_limits<Real>::epsilon());
  const Real invMagI = 1/magI, invMagJ = 1/magJ;

  for(int i=0;i<2;++i) {
    // also take absolute value since thats what we need when doing intersection checks later
    normalI[i]=std::fabs(normalI[i])*invMagI;
    normalJ[i]=std::fabs(normalJ[i])*invMagJ;
  }

  assert(normalI[0]>=0 && normalI[1]>=0);
  assert(normalJ[0]>=0 && normalJ[1]>=0);

  // Find the x,y,z max extents in lab frame ( exploit normal(I,J,K)[:] >=0 )
  const Real widthXvec[] = {w[0]*normalI[0], w[0]*normalI[1]};
  const Real widthYvec[] = {w[1]*normalJ[0], w[1]*normalJ[1]};

  for(int i=0; i<2; ++i) {
    objBoxLabFr[i][0] = c[i] -widthXvec[i] -widthYvec[i];
    objBoxLabFr[i][1] = c[i] +widthXvec[i] +widthYvec[i];
    objBoxObjFr[i][0] = c[i] -w[i];
    objBoxObjFr[i][1] = c[i] +w[i];
  }
}

bool AreaSegment::isIntersectingWithAABB(const Real start[2],const Real end[2]) const
{
  // Remember: Incoming coordinates are cell centers, not cell faces
  //start and end are two diagonally opposed corners of grid block
  // GN halved the safety here but added it back to w[] in prepare
  const Real AABB_w[2] = { //half block width + safe distance
      (end[0] - start[0])/2 + safe_distance,
      (end[1] - start[1])/2 + safe_distance
  };

  const Real AABB_c[2] = { //block center
    (end[0] + start[0])/2, (end[1] + start[1])/2
  };

  const Real AABB_box[2][2] = {
    {AABB_c[0] - AABB_w[0],  AABB_c[0] + AABB_w[0]},
    {AABB_c[1] - AABB_w[1],  AABB_c[1] + AABB_w[1]}
  };

  assert(AABB_w[0]>0 && AABB_w[1]>0);

  // Now Identify the ones that do not intersect
  Real intersectionLabFrame[2][2] = {
  {max(objBoxLabFr[0][0],AABB_box[0][0]),min(objBoxLabFr[0][1],AABB_box[0][1])},
  {max(objBoxLabFr[1][0],AABB_box[1][0]),min(objBoxLabFr[1][1],AABB_box[1][1])}
  };

  if ( intersectionLabFrame[0][1] - intersectionLabFrame[0][0] < 0
    || intersectionLabFrame[1][1] - intersectionLabFrame[1][0] < 0 )
    return false;

  // This is x-width of box, expressed in fish frame
  const Real widthXbox[2] = {AABB_w[0]*normalI[0], AABB_w[0]*normalJ[0]};
  // This is y-width of box, expressed in fish frame
  const Real widthYbox[2] = {AABB_w[1]*normalI[1], AABB_w[1]*normalJ[1]};

  const Real boxBox[2][2] = {
    { AABB_c[0] -widthXbox[0] -widthYbox[0],
      AABB_c[0] +widthXbox[0] +widthYbox[0]},
    { AABB_c[1] -widthXbox[1] -widthYbox[1],
      AABB_c[1] +widthXbox[1] +widthYbox[1]}
  };

  Real intersectionFishFrame[2][2] = {
   {max(boxBox[0][0],objBoxObjFr[0][0]), min(boxBox[0][1],objBoxObjFr[0][1])},
   {max(boxBox[1][0],objBoxObjFr[1][0]), min(boxBox[1][1],objBoxObjFr[1][1])}
  };

  if ( intersectionFishFrame[0][1] - intersectionFishFrame[0][0] < 0
    || intersectionFishFrame[1][1] - intersectionFishFrame[1][0] < 0)
    return false;

  return true;
}

void PutFishOnBlocks::operator()(const BlockInfo& i, FluidBlock& b,
  ObstacleBlock* const o, const vector<AreaSegment*>& v) const
{
  //std::chrono::time_point<std::chrono::high_resolution_clock> t0, t1, t2, t3;
  //t0 = std::chrono::high_resolution_clock::now();
  constructSurface(i, b, o, v);
  //t1 = std::chrono::high_resolution_clock::now();
  constructInternl(i, b, o, v);
  //t2 = std::chrono::high_resolution_clock::now();
  signedDistanceSqrt(i, b, o, v);
  //t3 = std::chrono::high_resolution_clock::now();
  //printf("%g %g %g\n",std::chrono::duration<Real>(t1-t0).count(),
  //                    std::chrono::duration<Real>(t2-t1).count(),
  //                    std::chrono::duration<Real>(t3-t2).count());
}

void PutFishOnBlocks::signedDistanceSqrt(const BlockInfo& info, FluidBlock& b, ObstacleBlock* const defblock, const vector<AreaSegment*>& vSegments) const
{
  // finalize signed distance function in tmpU
  const Real eps = std::numeric_limits<Real>::epsilon();
  for(int iy=0; iy<FluidBlock::sizeY; iy++)
  for(int ix=0; ix<FluidBlock::sizeX; ix++) {
    const Real normfac = b(ix,iy).tmpU > eps ? b(ix,iy).tmpU : 1;
    defblock->udef[iy][ix][0] /= normfac;
    defblock->udef[iy][ix][1] /= normfac;
    // change from signed squared distance function to normal sdf
    b(ix,iy).tmpU = defblock->chi[iy][ix] > 0 ?
      std::sqrt( defblock->chi[iy][ix]) : -std::sqrt(-defblock->chi[iy][ix]);
    //b(ix,iy,iz).tmpV = defblock->udef[iz][iy][ix][0]; //for debug
    //b(ix,iy,iz).tmpW = defblock->udef[iz][iy][ix][1]; //for debug
  }
}

void PutFishOnBlocks::constructSurface(const BlockInfo& info, FluidBlock& b, ObstacleBlock* const defblock, const vector<AreaSegment*>& vSegments) const
{
  Real org[2];
  info.pos(org, 0, 0);
  const Real h = info.h_gridpoint, invh = 1.0/info.h_gridpoint;
  const Real* const rX = cfish.rX;
  const Real* const rY = cfish.rY;
  const Real* const norX = cfish.norX;
  const Real* const norY = cfish.norY;
  const Real* const vX = cfish.vX;
  const Real* const vY = cfish.vY;
  const Real* const vNorX = cfish.vNorX;
  const Real* const vNorY = cfish.vNorY;
  const Real* const width = cfish.width;
  static constexpr int BS[2] = {FluidBlock::sizeX, FluidBlock::sizeY};
  std::fill(defblock->chi[0], defblock->chi[0] + BS[1]*BS[0], -1);

  // construct the shape (P2M with min(distance) as kernel) onto defblocks
  for(int i=0; i<(int)vSegments.size(); ++i) {
    //iterate over segments contained in the vSegm intersecting this block:
    const int firstSegm = max(vSegments[i]->s_range.first,           1);
    const int lastSegm =  min(vSegments[i]->s_range.second, cfish.Nm-2);
    for(int ss=firstSegm; ss<=lastSegm; ++ss) {
      assert(width[ss]>0);
      //for each segment, we have one point to left and right of midl
      for(int signp = -1; signp <= 1; signp+=2)
      {
        // create a surface point
        // special treatment of tail (width = 0 --> no ellipse, just line)
        Real myP[2]= { rX[ss+0] +width[ss+0]*signp*norX[ss+0],
                       rY[ss+0] +width[ss+0]*signp*norY[ss+0]  };
        Real pP[2] = { rX[ss+1] +width[ss+1]*signp*norX[ss+1],
                       rY[ss+1] +width[ss+1]*signp*norY[ss+1]  };
        Real pM[2] = { rX[ss-1] +width[ss-1]*signp*norX[ss-1],
                       rY[ss-1] +width[ss-1]*signp*norY[ss-1]  };
        changeToComputationalFrame(myP);
        changeToComputationalFrame(pP);
        changeToComputationalFrame(pM);
        const int iap[2] = {  (int)std::floor((myP[0]-org[0])*invh),
                              (int)std::floor((myP[1]-org[1])*invh) };
        Real udef[2] = { vX[ss+0] +width[ss+0]*signp*vNorX[ss+0],
                         vY[ss+0] +width[ss+0]*signp*vNorY[ss+0]    };
        changeVelocityToComputationalFrame(udef);
        // support is two points left, two points right --> Towers Chi will be one point left, one point right, but needs SDF wider
        for(int sy =std::max(0, iap[1]-1); sy <std::min(iap[1]+3, BS[1]); ++sy)
        for(int sx =std::max(0, iap[0]-1); sx <std::min(iap[0]+3, BS[0]); ++sx)
        {
          Real p[2];
          info.pos(p, sx, sy);
          const Real dist0 = eulerDistSq2D(p, myP);
          const Real distP = eulerDistSq2D(p,  pP);
          const Real distM = eulerDistSq2D(p,  pM);

          if(std::fabs(defblock->chi[sy][sx])<std::min({dist0,distP,distM}))
            continue;

          changeFromComputationalFrame(p);
          #ifndef NDEBUG // check that change of ref frame does not affect dist
            const Real p0[2] = {rX[ss] +width[ss]*signp*norX[ss],
                                rY[ss] +width[ss]*signp*norY[ss] };
            const Real distC = eulerDistSq2D(p, p0);
            assert(std::fabs(distC-dist0)<EPS);
          #endif

          int close_s = ss, secnd_s = ss + (distP<distM? 1 : -1);
          Real dist1 = dist0, dist2 = distP<distM? distP : distM;
          if(distP < dist0 || distM < dist0) { // switch nearest surf point
            dist1 = dist2; dist2 = dist0;
            close_s = secnd_s; secnd_s = ss;
          }

          const Real dSsq = std::pow(rX[close_s]-rX[secnd_s], 2)
                           +std::pow(rY[close_s]-rY[secnd_s], 2);
          assert(dSsq > 2.2e-16);
          const Real cnt2ML = std::pow( width[close_s],2);
          const Real nxt2ML = std::pow( width[secnd_s],2);
          const Real safeW = std::max( width[close_s], width[secnd_s] ) + 2*h;
          const Real xMidl[2] = {rX[close_s], rY[close_s]};
          const Real grd2ML = eulerDistSq2D(p, xMidl);
          const Real diffH = std::fabs( width[close_s] - width[secnd_s] );
          Real sign2d = 0;
          //If width changes slowly or if point is very far away, this is safer:
          if( dSsq > diffH*diffH || grd2ML > safeW*safeW )
          { // if no abrupt changes in width we use nearest neighbour
            sign2d = grd2ML > cnt2ML ? -1 : 1;
          }
          else
          {
            // else we model the span between ellipses as a spherical segment
            // http://mathworld.wolfram.com/SphericalSegment.html
            const Real corr = 2*std::sqrt(cnt2ML*nxt2ML);
            const Real Rsq = (cnt2ML +nxt2ML -corr +dSsq) //radius of the sphere
                            *(cnt2ML +nxt2ML +corr +dSsq)/4/dSsq;
            const Real maxAx = std::max(cnt2ML, nxt2ML);
            const int idAx1 = cnt2ML> nxt2ML? close_s : secnd_s;
            const int idAx2 = idAx1==close_s? secnd_s : close_s;
            // 'submerged' fraction of radius:
            const Real d = std::sqrt((Rsq - maxAx)/dSsq); // (divided by ds)
            // position of the centre of the sphere:
            const Real xCentr[2] = {rX[idAx1] +(rX[idAx1]-rX[idAx2])*d,
                                   rY[idAx1] +(rY[idAx1]-rY[idAx2])*d};
            const Real grd2Core = eulerDistSq2D(p, xCentr);
            sign2d = grd2Core > Rsq ? -1 : 1; // as always, neg outside
          }

          if(std::fabs(defblock->chi[sy][sx]) > dist1) {
            defblock->udef[sy][sx][0] = udef[0];
            defblock->udef[sy][sx][1] = udef[1];
            defblock->chi [sy][sx] = sign2d*dist1;
            b(sx,sy).tmpU = 1;
          }
          // Not chi yet, I stored squared distance from analytical boundary
          // distSq is updated only if curr value is smaller than the old one
        }
      }
    }
  }
}

void PutFishOnBlocks::constructInternl(const BlockInfo& info, FluidBlock& b, ObstacleBlock* const defblock, const vector<AreaSegment*>& vSegments) const
{
  Real org[2];
  info.pos(org, 0, 0);
  const Real h = info.h_gridpoint, invh = 1.0/info.h_gridpoint;
  // construct the deformation velocities (P2M with hat function as kernel)
  for(int i=0; i<(int)vSegments.size(); ++i)
  {
  const int firstSegm = max(vSegments[i]->s_range.first,           1);
  const int lastSegm =  min(vSegments[i]->s_range.second, cfish.Nm-2);
  for(int ss=firstSegm; ss<=lastSegm; ++ss)
  {
    // P2M udef of a slice at this s
    const Real myWidth = cfish.width[ss];
    assert(myWidth > 0);
    //here we process also all inner points. Nw to the left and right of midl
    // add xtension here to make sure we have it in each direction:
    const int Nw = std::floor(myWidth/h); //floor bcz we already did interior
    for(int iw = -Nw+1; iw < Nw; ++iw)
    {
      const Real offsetW = iw * h;
      Real xp[2] = { cfish.rX[ss] + offsetW*cfish.norX[ss],
                     cfish.rY[ss] + offsetW*cfish.norY[ss] };
      changeToComputationalFrame(xp);
      xp[0] = (xp[0]-org[0])*invh; // how many grid points from this block
      xp[1] = (xp[1]-org[1])*invh; // origin is this fishpoint located at?
      Real udef[2] = { cfish.vX[ss] + offsetW*cfish.vNorX[ss],
                       cfish.vY[ss] + offsetW*cfish.vNorY[ss] };
      changeVelocityToComputationalFrame(udef);
      const Real ap[2] = { std::floor(xp[0]), std::floor(xp[1]) };
      const int iap[2] = { (int)ap[0], (int)ap[1] };
      Real wghts[2][2]; // P2M weights
      for(int c=0; c<2; ++c) {
        const Real t[2] = { // we floored, hat between xp and grid point +-1
            std::fabs(xp[c] -ap[c]), std::fabs(xp[c] -(ap[c] +1))
        };
        wghts[c][0] = 1 - t[0];
        wghts[c][1] = 1 - t[1];
      }
      for(int sy=max(0,0-iap[1]); sy<min(2,FluidBlock::sizeY-iap[1]); ++sy)
      for(int sx=max(0,0-iap[0]); sx<min(2,FluidBlock::sizeX-iap[0]); ++sx) {
        const Real wxwy = wghts[1][sy] * wghts[0][sx];
        const int idx = iap[0]+sx, idy = iap[1]+sy;
        assert(idx>=0 && idx<FluidBlock::sizeX);
        assert(idy>=0 && idy<FluidBlock::sizeY);
        assert(wxwy>=0 && wxwy<=1);
        defblock->udef[idy][idx][0] += wxwy*udef[0];
        defblock->udef[idy][idx][1] += wxwy*udef[1];
        b(idx,idy).tmpU += wxwy;

        // set sign for all interior points
        if( std::fabs(defblock->chi[idy][idx]+1)<EPS )
          defblock->chi[idy][idx] = 1;
      }
    }
  }
  }
}

void PutFishOnBlocks_Finalize::operator()(Lab & lab, const BlockInfo& info,
  FluidBlock&b, ObstacleBlock* const defblock) const
{
  const Real h = info.h_gridpoint, i2h = 0.5/h;//, fac = 0.5*h;
  static constexpr Real EPS = std::numeric_limits<Real>::epsilon();
  for(int iy=0; iy<FluidBlock::sizeY; iy++)
  for(int ix=0; ix<FluidBlock::sizeX; ix++) {
    Real p[2];
    info.pos(p, ix,iy);
    const auto U = defblock->udef[iy][ix][0], V = defblock->udef[iy][ix][1];
    if (lab(ix,iy).tmpU > +2*h || lab(ix,iy).tmpU < -2*h) {
      const Real H = lab(ix,iy).tmpU > 0 ? 1 : 0;
      b(ix,iy).tmp = std::max(H, b(ix,iy).tmp);
      defblock->write(ix, iy, U, V, 1, H, 0, 0, 0, h);
      continue;
    }

    const Real distPx = lab(ix+1,iy).tmpU, distMx = lab(ix-1,iy).tmpU;
    const Real distPy = lab(ix,iy+1).tmpU, distMy = lab(ix,iy-1).tmpU;
    const Real IplusX = distPx<0? 0 : distPx, IminuX = distMx<0? 0 : distMx;
    const Real IplusY = distPy<0? 0 : distPy, IminuY = distMy<0? 0 : distMy;
    const Real HplusX = std::fabs(distPx)<EPS? (Real).5 : ( distPx<0? 0 : 1 );
    const Real HminuX = std::fabs(distMx)<EPS? (Real).5 : ( distMx<0? 0 : 1 );
    const Real HplusY = std::fabs(distPy)<EPS? (Real).5 : ( distPy<0? 0 : 1 );
    const Real HminuY = std::fabs(distMy)<EPS? (Real).5 : ( distMy<0? 0 : 1 );

    const Real gradIX = i2h*(IplusX-IminuX), gradIY = i2h*(IplusY-IminuY);
    const Real gradUX = i2h*(distPx-distMx), gradUY = i2h*(distPy-distMy);
    const Real gradHX =     (HplusX-HminuX), gradHY =     (HplusY-HminuY);

    const Real gradUSq = gradUX*gradUX + gradUY*gradUY;
    const Real denum = gradUSq<EPS? EPS : gradUSq;
    const Real H     =     ((gradIX*gradUX + gradIY*gradUY)/denum);
    const Real Delta =     ((gradHX*gradUX + gradHY*gradUY)/denum);
    defblock->write(ix, iy, U, V, 1, H, Delta, gradUX, gradUY, h);
    b(ix,iy).tmp = std::max(H, b(ix,iy).tmp);
  }
}
