//
//  Contains definitions of:
// - cylinder
// - half cylinder
// - ellipse (TODO half ellipse?)
// - plate, allows imposed angular velocity (eg, fins)
// - cylinder with two densities
// - ellipse with two densities.
// All done with Tower's discretized heaviside chi and dirac delta
// Ellipse, due to computational cost, requires first running FillBlocks_Ellipse
// on all blocks, which writes onto tmp, then either EllipseFinalize or VarRho

#include "ShapeLibrary.h"

#define towersDeltaH(inv2h,distPx,distMx,distPy,distMy,H,Delta,gradUX,gradUY)\
do {                                                                         \
  static const Real eps = std::numeric_limits<Real>::epsilon();              \
  const Real IplusX = distPx < 0 ? 0 : distPx;                               \
  const Real IminuX = distMx < 0 ? 0 : distMx;                               \
  const Real IplusY = distPy < 0 ? 0 : distPy;                               \
  const Real IminuY = distMy < 0 ? 0 : distMy;                               \
  const Real HplusX = distPx == 0 ? 0.5 : (distPx < 0 ? 0 : 1);              \
  const Real HminuX = distMx == 0 ? 0.5 : (distMx < 0 ? 0 : 1);              \
  const Real HplusY = distPy == 0 ? 0.5 : (distPy < 0 ? 0 : 1);              \
  const Real HminuY = distMy == 0 ? 0.5 : (distMy < 0 ? 0 : 1);              \
  /* all would be multiplied by 0.5/h, simplifies out later */               \
  const Real gradIX = inv2h*(IplusX-IminuX), gradIY = inv2h*(IplusY-IminuY); \
  const Real gradHX =       (HplusX-HminuX), gradHY =       (HplusY-HminuY); \
             gradUX = inv2h*(distPx-distMx); gradUY = inv2h*(distPy-distMy); \
  const Real gradUSq = gradUX*gradUX + gradUY*gradUY + eps;                  \
  const Real numH    = gradIX*gradUX + gradIY*gradUY;                        \
  const Real numD    = gradHX*gradUX + gradHY*gradUY;                        \
  Delta = numD / gradUSq; H = numH / gradUSq;                                \
} while(0)
//  Delta   = gradUSq < eps ? 0 : numD / gradUSq;
//  H       = gradUSq < eps ? 0 : numH / gradUSq;

void FillBlocks_Cylinder::operator()(const BlockInfo& info, FluidBlock& block, ObstacleBlock * const obstblock) const
{
  const Real h = info.h_gridpoint;
  const Real inv2h = (Real)0.5 / h;
  if(_is_touching(info))
  for(int iy=0; iy<FluidBlock::sizeY; iy++)
  for(int ix=0; ix<FluidBlock::sizeX; ix++)
  {
    Real p[2], H=0, Delta=0, gradUX=0, gradUY=0;
    info.pos(p, ix, iy);
    p[0] -= cylinder_position[0];
    p[1] -= cylinder_position[1];

    const Real dist = distanceTocylinder(p[0],p[1]);
    if(dist > 2*h || dist < -2*h) { //2 should be safe
      H = dist > 0 ? 1.0 : 0.0;
    } else {
      const Real distPx = distanceTocylinder(p[0]+h,p[1]);
      const Real distMx = distanceTocylinder(p[0]-h,p[1]);
      const Real distPy = distanceTocylinder(p[0],p[1]+h);
      const Real distMy = distanceTocylinder(p[0],p[1]-h);
      towersDeltaH(inv2h,distPx,distMx,distPy,distMy,H,Delta,gradUX,gradUY);
    }

    #ifndef NDEBUG
    if (H < obstblock->chi[iy][ix]) {
      printf("FillBlocks_Cylinder: Error is obstblock->chi \n");
      fflush(0); abort();
    }
    #endif

    obstblock->write(ix, iy, rhoS, H, Delta, gradUX, gradUY, h);
    block(ix,iy).invRho = H / rhoS + block(ix, iy).invRho*(1-H);
  }
}

void FillBlocks_HalfCylinder::operator()(const BlockInfo& info, FluidBlock& block, ObstacleBlock * const obstblock) const
{
  const Real h = info.h_gridpoint;
  const Real inv2h = (Real)0.5 / h;
  //if(_is_touching(info)) TODO THERE IS A BUG THERE
  for(int iy=0; iy<FluidBlock::sizeY; iy++)
  for(int ix=0; ix<FluidBlock::sizeX; ix++)
  {
    Real p[2], H=0, Delta=0, gradUX=0, gradUY=0;
    info.pos(p, ix, iy);
    p[0] -= cylinder_position[0]; p[1] -= cylinder_position[1];

    const Real dist = distanceTocylinder(p[0],p[1]);
    if(dist > 2*h || dist < -2*h) { //2 should be safe
      H = dist > 0 ? 1.0 : 0.0;
    } else {
      const Real distPx = distanceTocylinder(p[0]+h,p[1]);
      const Real distMx = distanceTocylinder(p[0]-h,p[1]);
      const Real distPy = distanceTocylinder(p[0],p[1]+h);
      const Real distMy = distanceTocylinder(p[0],p[1]-h);
      towersDeltaH(inv2h,distPx,distMx,distPy,distMy,H,Delta,gradUX,gradUY);
    }

    #ifndef NDEBUG
    if ( H > 0 && H < obstblock->chi[iy][ix]) {
      printf("FillBlocks_HalfCylinder: Error is obstblock->chi \n");
      fflush(0); abort();
    }
    #endif

    obstblock->write(ix, iy, rhoS, H, Delta, gradUX, gradUY, h);
    block(ix,iy).invRho = H / rhoS + block(ix, iy).invRho*(1-H);
  }
}

Real FillBlocks_Ellipse::DistancePointEllipseSpecial (const Real e[2], const Real y[2], Real x[2])
{
  static constexpr int imax = 2*std::numeric_limits<Real>::max_exponent;
  static constexpr Real eps = std::numeric_limits<Real>::epsilon();
  if (y[1] > (Real)0) {
    if (y[0] > (Real)0) {
      // Bisect to compute the root of F(t) for t >= -e1*e1.
      const Real esqr[2] = { e[0]*e[0], e[1]*e[1] };
      const Real ey[2] = { e[0]*y[0], e[1]*y[1] };
      Real t0 = -esqr[1] + ey[1];
      Real t1 = -esqr[1] + sqrt(ey[0]*ey[0] + ey[1]*ey[1]);
      Real t = t0;
      for (int i = 0; i < imax; ++i) {
        t = ((Real)0.5)*(t0 + t1);
        if ( std::fabs(t-t0)<eps || std::fabs(t-t1)<eps ) break;

        const Real r[2] = { ey[0]/(t + esqr[0]), ey[1]/(t + esqr[1]) };
        const Real f = r[0]*r[0] + r[1]*r[1] - (Real)1;
        if (f > (Real)0) t0 = t;
        else if (f < (Real)0) t1 = t;
        else break;
      }

      x[0] = esqr[0]*y[0]/(t + esqr[0]);
      x[1] = esqr[1]*y[1]/(t + esqr[1]);
      const Real d[2] = { x[0] - y[0], x[1] - y[1] };
      return std::sqrt(d[0]*d[0] + d[1]*d[1]);
    } else { // y0 == 0
      x[0] = (Real)0;
      x[1] = e[1];
      return std::fabs(y[1] - e[1]);
    }
  } else { // y1 == 0
    const Real denom0 = e[0]*e[0] - e[1]*e[1];
    const Real e0y0 = e[0]*y[0];
    if (e0y0 < denom0) {
      // y0 is inside the subinterval.
      const Real x0de0 = e0y0/denom0;
      const Real x0de0sqr = x0de0*x0de0;
      x[0] = e[0]*x0de0;
      x[1] = e[1]*std::sqrt(std::fabs((Real)1 - x0de0sqr));
      const Real d0 = x[0] - y[0];
      return std::sqrt(d0*d0 + x[1]*x[1]);
    } else {
      // y0 is outside the subinterval.  The closest ellipse point has
      // x1 == 0 and is on the domain-boundary interval (x0/e0)^2 = 1.
      x[0] = e[0];
      x[1] = (Real)0;
      return std::fabs(y[0] - e[0]);
    }
  }
}
//----------------------------------------------------------------------------
// The ellipse is (x0/e0)^2 + (x1/e1)^2 = 1.  The query point is (y0,y1).
// The function returns the distance from the query point to the ellipse.
// It also computes the ellipse point (x0,x1) that is closest to (y0,y1).
//----------------------------------------------------------------------------

Real FillBlocks_Ellipse::DistancePointEllipse (const Real e[2], const Real y[2], Real x[2])
{
  // Determine reflections for y to the first quadrant.
  bool reflect[2];
  for (int i = 0; i < 2; ++i) reflect[i] = (y[i] < (Real)0);

  // Determine the axis order for decreasing extents.
  int permute[2];
  if (e[0] < e[1]) { permute[0] = 1;  permute[1] = 0; }
  else { permute[0] = 0;  permute[1] = 1; }

  int invpermute[2];
  for (int i = 0; i < 2; ++i) invpermute[permute[i]] = i;

  Real locE[2], locY[2];
  for (int i = 0; i < 2; ++i) {
    const int j = permute[i];
    locE[i] = e[j];
    locY[i] = y[j];
    if (reflect[j]) locY[i] = -locY[i];
  }

  Real locX[2];
  const Real distance = DistancePointEllipseSpecial(locE, locY, locX);

  // Restore the axis order and reflections.
  for (int i = 0; i < 2; ++i) {
    const int j = invpermute[i];
    if (reflect[j]) locX[j] = -locX[j];
    x[i] = locX[j];
  }

  return distance;
}

void FillBlocks_Ellipse::operator()(const BlockInfo& info, FluidBlock& block, ObstacleBlock * const obstblock) const
{
  const Real e[2] = {e0,e1};
  const Real sqMinSemiAx = e[0]>e[1] ? e[1]*e[1] : e[0]*e[0];

  if(_is_touching(info))
  {
    for(int iy=0; iy<FluidBlock::sizeY; iy++)
    for(int ix=0; ix<FluidBlock::sizeX; ix++)
    {
      Real p[2], xs[2];
      info.pos(p, ix, iy);
      p[0] -= position[0]; p[1] -= position[1];
      const Real t[2] = {cosang*p[0]-sinang*p[1], sinang*p[0]+cosang*p[1]};
      const Real sqDist = p[0]*p[0] + p[1]*p[1];

      if (fabs(t[0]) > e[0]+safety || fabs(t[1]) > e[1]+safety )
        block(ix, iy).tmp = -1; //is outside
      else if (sqDist + safety*safety < sqMinSemiAx)
        block(ix, iy).tmp =  1; //is inside
      else {
        const Real dist = DistancePointEllipse (e, t, xs);
        const int sign = sqDist > (xs[0]*xs[0]+xs[1]*xs[1]) ? -1 : 1;
        block(ix, iy).tmp = sign*dist;
      }
    }
  }
}

void FillBlocks_EllipseFinalize::operator()(Lab& l, const BlockInfo&i, FluidBlock&b, ObstacleBlock*const o) const
{
  const Real inv2h = (Real)0.5 / h;
  for(int iy=0; iy<FluidBlock::sizeY; iy++)
  for(int ix=0; ix<FluidBlock::sizeX; ix++)
  {
    Real H=0, Delta=0, gradUX=0, gradUY=0;

    const Real dist = l(ix,iy).tmp;
    if(dist > 2*h || dist < -2*h) { //2 should be safe
      H = dist > 0 ? 1.0 : 0.0;
    } else {
      const Real distPx = l(ix+1,iy).tmp, distMx = l(ix-1,iy).tmp;
      const Real distPy = l(ix,iy+1).tmp, distMy = l(ix,iy-1).tmp;
      towersDeltaH(inv2h,distPx,distMx,distPy,distMy,H,Delta,gradUX,gradUY);
    }

    #ifndef NDEBUG
    if (H > 0 && H < o->chi[iy][ix]) {
      printf("FillBlocks_Ellipse: Error is obstblock->chi \n");
      fflush(0); abort();
    }
    #endif

    o->write(ix, iy, rhoS, H, Delta, gradUX, gradUY, h);
    b(ix,iy).invRho = H / rhoS + b(ix, iy).invRho*(1-H);
  }
}

void FillBlocks_Plate::operator()(const BlockInfo& info, FluidBlock& block, ObstacleBlock * const obstblock) const
{
  const Real h = info.h_gridpoint;
  const Real inv2h = (Real)0.5 / h;
  if(_is_touching(info))
  for(int iy=0; iy<FluidBlock::sizeY; iy++)
  for(int ix=0; ix<FluidBlock::sizeX; ix++)
  {
    Real p[2], H=0, Delta=0, gradUX=0, gradUY=0;
    info.pos(p, ix, iy);
    p[0] -= position[0]; p[1] -= position[1];
    const Real udef = -p[1]*angvel, vdef = p[0]*angvel;

    const Real dist = distance(p[0], p[1]);
    if(dist > 2*h || dist < -2*h) { //2 should be safe
      H = dist > 0 ? 1.0 : 0.0;
    } else {
      const Real distPx=distance(p[0]+h,p[1]), distMx=distance(p[0]-h,p[1]);
      const Real distPy=distance(p[0],p[1]+h), distMy=distance(p[0],p[1]-h);
      towersDeltaH(inv2h,distPx,distMx,distPy,distMy,H,Delta,gradUX,gradUY);
    }

    #ifndef NDEBUG
    if (H > 0 && H < obstblock->chi[iy][ix]) {
      printf("FillBlocks_Plate: Error is obstblock->chi \n");
      fflush(0); abort();
    }
    #endif
    obstblock->write(ix, iy, udef, vdef, rhoS, H, Delta, gradUX, gradUY, h);
    block(ix,iy).invRho = H / rhoS + block(ix, iy).invRho*(1-H);
  }
}

void FillBlocks_VarRhoCylinder::operator()(const BlockInfo& info, FluidBlock& block, ObstacleBlock * const obstblock) const
{
  const Real h = info.h_gridpoint;
  const Real inv2h = (Real)0.5 / h;
  if(_is_touching(info))
  for(int iy=0; iy<FluidBlock::sizeY; iy++)
  for(int ix=0; ix<FluidBlock::sizeX; ix++)
  {
    Real p[2], H=0, Delta=0, gradUX=0, gradUY=0;
    info.pos(p, ix, iy);
    p[0] -= cylinder_position[0];
    p[1] -= cylinder_position[1];

    const Real dist = distanceTocylinder(p[0],p[1]);
    if(dist > 2*h || dist < -2*h) { //2 should be safe
      H = dist > 0 ? 1.0 : 0.0;
    } else {
      const Real distPx = distanceTocylinder(p[0]+h,p[1]);
      const Real distMx = distanceTocylinder(p[0]-h,p[1]);
      const Real distPy = distanceTocylinder(p[0],p[1]+h);
      const Real distMy = distanceTocylinder(p[0],p[1]-h);
      towersDeltaH(inv2h,distPx,distMx,distPy,distMy,H,Delta,gradUX,gradUY);
    }

    #ifndef NDEBUG
    if (H > 0 && H < obstblock->chi[iy][ix]) {
      printf("FillBlocks_VarRhoCylinder: Error is obstblock->chi \n");
      fflush(0); abort();
    }
    #endif

    //const Real x =  p[0]*cosang + p[1]*sinang;
    const Real y = -p[0]*sinang + p[1]*cosang;
    const Real Y = 0.5*y/h; //>0 is top, <0 is bottom
    const Real moll = Y>1 ? 0 : (Y<-1 ? 1 : mollified_heaviside(Y));
    //const Real moll = Y>0 ? 0 : 1;
    const Real rhoS = (1-moll)*rhoTop + moll*rhoBot;
    obstblock->write(ix, iy, rhoS, H, Delta, gradUX, gradUY, h);
    block(ix,iy).invRho = H / rhoS + block(ix, iy).invRho*(1-H);
  }
}

void FillBlocks_VarRhoEllipseFinalize::operator()(Lab& l, const BlockInfo&i, FluidBlock&b, ObstacleBlock*const o) const
{
  const Real inv2h = (Real)0.5 / h;
  for(int iy=0; iy<FluidBlock::sizeY; iy++)
  for(int ix=0; ix<FluidBlock::sizeX; ix++)
  {
    Real p[2], H=0, Delta=0, gradUX=0, gradUY=0;
    i.pos(p, ix, iy);
    p[0] -= position[0];
    p[1] -= position[1];

    const Real dist = l(ix,iy).tmp;
    if(dist > 2*h || dist < -2*h) { //2 should be safe
      H = dist > 0 ? 1.0 : 0.0;
    } else {
      const Real distPx = l(ix+1,iy).tmp, distMx = l(ix-1,iy).tmp;
      const Real distPy = l(ix,iy+1).tmp, distMy = l(ix,iy-1).tmp;
      towersDeltaH(inv2h,distPx,distMx,distPy,distMy,H,Delta,gradUX,gradUY);
    }

    #ifndef NDEBUG
    if (H < o->chi[iy][ix]) {
      printf("FillBlocks_VarRhoEllipseFinalize: Error is obstblock->chi \n");
      fflush(0); abort();
    }
    #endif

    const Real y = -p[0]*sinang + p[1]*cosang;
    const Real Y = 0.5*y/h; //>0 is top, <0 is bottom
    const Real moll = Y>1 ? 0 : (Y<-1 ? 1 : mollified_heaviside(Y));
    const Real rhoS = (1-moll)*rhoTop + moll*rhoBot;
    o->write(ix, iy, rhoS, H, Delta, gradUX, gradUY, h);
    b(ix,iy).invRho = H / rhoS + b(ix, iy).invRho*(1-H);
  }
}
