//
//  CubismUP_2D
//  Copyright (c) 2018 CSE-Lab, ETH Zurich, Switzerland.
//  Distributed under the terms of the MIT license.
//
//  Created by Guido Novati (novatig@ethz.ch).
//


#include "ShapeLibrary.h"

static inline Real mollified_heaviside(const Real x) {
  const Real alpha = M_PI * std::min( (Real)1, std::max( (Real)0, (x+1)/2 ) );
  return 0.5 + 0.5 * std::cos( alpha );
}

static Real distPointEllipseSpecial(const Real e[2],const Real y[2],Real x[2]);
static Real distPointEllipse(const Real e[2], const Real y[2], Real x[2]);

void FillBlocks_Cylinder::operator()(const BlockInfo& I,
                                         ScalarBlock& B,
                                       ObstacleBlock& O) const
{
  if( _is_touching(I, bbox, safety) )
  for(int iy=0; iy<ObstacleBlock::sizeY; iy++)
  for(int ix=0; ix<ObstacleBlock::sizeX; ix++)
  {
    Real p[2]; I.pos(p, ix, iy); p[0] -= pos[0]; p[1] -= pos[1];
    B(ix,iy).s = std::max( B(ix,iy).s, distanceTocylinder(p[0], p[1]) );
    //B(ix,iy).invRho = H / rhoS + block(ix, iy).invRho*(1-H);
  }
}

void FillBlocks_HalfCylinder::operator()(const BlockInfo& I,
                                             ScalarBlock& B,
                                           ObstacleBlock& O) const
{
  if( _is_touching(I, bbox, safety) )
  {
    for(int iy=0; iy<ObstacleBlock::sizeY; iy++)
    for(int ix=0; ix<ObstacleBlock::sizeX; ix++)
    {
      Real p[2]; I.pos(p, ix, iy); p[0] -= pos[0]; p[1] -= pos[1];
      B(ix,iy).s = std::max( B(ix,iy).s, distanceTocylinder(p[0], p[1]) );
      //B(ix,iy).invRho = H / rhoS + block(ix, iy).invRho*(1-H);
    }
  }
}

void FillBlocks_VarRhoCylinder::operator()(const BlockInfo& I,
                                               ScalarBlock& B,
                                             ObstacleBlock& O) const
{
  if( _is_touching(I, bbox, safety) )
  {
    for(int iy=0; iy<ObstacleBlock::sizeY; iy++)
    for(int ix=0; ix<ObstacleBlock::sizeX; ix++)
    {
      Real p[2]; I.pos(p, ix, iy); p[0] -= pos[0]; p[1] -= pos[1];
      B(ix,iy).s = std::max( B(ix,iy).s, distanceTocylinder(p[0], p[1]) );
      //const Real x =  p[0]*cosang + p[1]*sinang;
      //const Real y = -p[0]*sinang + p[1]*cosang;
      //const Real Y = 0.5*y/h; //>0 is top, <0 is bottom
      //const Real moll = Y>1 ? 0 : (Y<-1 ? 1 : mollified_heaviside(Y));
      //const Real moll = Y>0 ? 0 : 1;
      //const Real rhoS = (1-moll)*rhoTop + moll*rhoBot;
      //block(ix,iy).invRho = H / rhoS + block(ix, iy).invRho*(1-H);
    }
  }
}

void FillBlocks_Plate::operator()(const BlockInfo& I,
                                      ScalarBlock& B,
                                    ObstacleBlock& O) const
{
  if( _is_touching(I, bbox, safety) )
  {
    for(int iy=0; iy<ObstacleBlock::sizeY; iy++)
    for(int ix=0; ix<ObstacleBlock::sizeX; ix++)
    {
      Real p[2]; I.pos(p, ix, iy); p[0] -= pos[0]; p[1] -= pos[1];
      O.udef[iy][ix][0] = -p[1]*angvel; O.udef[iy][ix][1] =  p[0]*angvel;
      B(ix,iy).s = std::max( B(ix,iy).s, distance(p[0], p[1]) );
    }
  }
}

void FillBlocks_Ellipse::operator()(const BlockInfo& I,
                                        ScalarBlock& B,
                                      ObstacleBlock& O) const
{
  if( _is_touching(I, bbox, safety) )
  {
    for(int iy=0; iy<ObstacleBlock::sizeY; iy++)
    for(int ix=0; ix<ObstacleBlock::sizeX; ix++)
    {
      Real p[2], xs[2];
      I.pos(p, ix, iy); p[0] -= pos[0]; p[1] -= pos[1];
      const Real t[2] = {cosang*p[0]-sinang*p[1], sinang*p[0]+cosang*p[1]};
      const Real sqDist = p[0]*p[0] + p[1]*p[1];
      Real dist = 0;
      if (std::fabs(t[0]) > e[0]+safety || std::fabs(t[1]) > e[1]+safety )
        dist = -1; //is outside
      else if (sqDist + safety*safety < sqMinSemiAx)
        dist =  1; //is inside
      else {
        const Real absdist = distPointEllipse (e, t, xs);
        const int sign = sqDist > (xs[0]*xs[0]+xs[1]*xs[1]) ? -1 : 1;
        dist = sign * absdist;
      }
      B(ix,iy).s = std::max( B(ix,iy).s, dist );
    }
  }
}

void FillBlocks_VarRhoEllipse::operator()(const BlockInfo& I,
                                              ScalarBlock& B,
                                            ObstacleBlock& O) const
{
  if( _is_touching(I, bbox, safety) )
  {
    for(int iy=0; iy<ObstacleBlock::sizeY; iy++)
    for(int ix=0; ix<ObstacleBlock::sizeX; ix++)
    {
      Real p[2], xs[2];
      I.pos(p, ix, iy); p[0] -= pos[0]; p[1] -= pos[1];
      const Real t[2] = {cosang*p[0]-sinang*p[1], sinang*p[0]+cosang*p[1]};
      const Real sqDist = p[0]*p[0] + p[1]*p[1];
      Real dist = 0;
      if (std::fabs(t[0]) > e[0]+safety || std::fabs(t[1]) > e[1]+safety )
        dist = -1; //is outside
      else if (sqDist + safety*safety < sqMinSemiAx)
        dist =  1; //is inside
      else {
        const Real absdist = distPointEllipse (e, t, xs);
        const int sign = sqDist > (xs[0]*xs[0]+xs[1]*xs[1]) ? -1 : 1;
        dist = sign * absdist;
      }
      B(ix,iy).s = std::max( B(ix,iy).s, dist );
    }
      //const Real y = -p[0]*sinang + p[1]*cosang;
      //const Real Y = 0.5 * y / h; //>0 is top, <0 is bottom
      //const Real moll = Y>1 ? 0 : (Y<-1 ? 1 : mollified_heaviside(Y));
      //const Real rhoS = (1-moll)*rhoTop + moll*rhoBot;
  }
}

Real distPointEllipseSpecial(const Real e[2], const Real y[2], Real x[2])
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

Real distPointEllipse(const Real e[2], const Real y[2], Real x[2])
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
  const Real distance = distPointEllipseSpecial(locE, locY, locX);

  // Restore the axis order and reflections.
  for (int i = 0; i < 2; ++i) {
    const int j = invpermute[i];
    if (reflect[j]) locX[j] = -locX[j];
    x[i] = locX[j];
  }

  return distance;
}
