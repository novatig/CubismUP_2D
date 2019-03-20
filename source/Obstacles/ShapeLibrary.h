//
//  CubismUP_2D
//  Copyright (c) 2018 CSE-Lab, ETH Zurich, Switzerland.
//  Distributed under the terms of the MIT license.
//
//  Created by Guido Novati (novatig@ethz.ch).
//

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

#pragma once

#include "../Definitions.h"
#include "../ObstacleBlock.h"

inline bool _is_touching(
  const BlockInfo& INFO, const Real BBOX[2][2], const Real safety )
{
  Real MINP[2], MAXP[2];
  INFO.pos(MINP, 0, 0);
  INFO.pos(MAXP, ObstacleBlock::sizeX-1, ObstacleBlock::sizeY-1);
  //for(int i=0; i<2; ++i) { MINP[i] -= safety; MAXP[i] += safety; }
  const Real intrsct[2][2] = {
   { std::max(MINP[0], BBOX[0][0]), std::min(MAXP[0], BBOX[0][1]) },
   { std::max(MINP[1], BBOX[1][0]), std::min(MAXP[1], BBOX[1][1]) }
  };
  return intrsct[0][1] - intrsct[0][0]>0 && intrsct[1][1] - intrsct[1][0]>0;
}

struct FillBlocks_Cylinder
{
  const Real radius, safety, rhoS, pos[2], bbox[2][2] = {
    { pos[0] - radius - safety, pos[0] + radius + safety },
    { pos[1] - radius - safety, pos[1] + radius + safety }
  };

  FillBlocks_Cylinder(Real R, Real h, double C[2], Real rho) :
    radius(R), safety(2*h), rhoS(rho), pos{(Real)C[0], (Real)C[1]} {}

  inline Real distanceTocylinder(const Real x, const Real y) const {
      return radius - std::sqrt(x*x+y*y); // pos inside, neg outside
  }

  inline bool is_touching(const BlockInfo& INFO) const {
    return _is_touching(INFO, bbox, safety);
  }

  void operator()(const BlockInfo& I, ScalarBlock& B, ObstacleBlock& O) const;
};

struct FillBlocks_HalfCylinder
{
  const Real radius, safety, pos[2], angle, rhoS;
  const Real cosang = std::cos(angle), sinang = std::sin(angle);
  const Real bbox[2][2] = {
    { pos[0] - radius - safety, pos[0] + radius + safety },
    { pos[1] - radius - safety, pos[1] + radius + safety }
  };

  FillBlocks_HalfCylinder(Real R, Real h, double C[2], Real rho, Real ang):
    radius(R), safety(2*h), pos{(Real)C[0],(Real)C[1]}, angle(ang), rhoS(rho) {}

  inline Real distanceTocylinder(const Real x, const Real y) const {
    const Real X =   x * cosang + y * sinang;
    if(X>0) return -X;
    //const Real Y = - x*sinang + y*cosang; /// For default orientation
    //if(Y>0) return -Y;                    /// pointing downwards.
    else return radius - std::sqrt(x*x+y*y); // (pos inside, neg outside)
  }

  inline bool is_touching(const BlockInfo& INFO) const {
    return _is_touching(INFO, bbox, safety);
  }

  void operator()(const BlockInfo& I, ScalarBlock& B, ObstacleBlock& O) const;
};

struct FillBlocks_Ellipse
{
  const Real e0, e1, safety, pos[2], angle, rhoS;
  const Real e[2] = {e0, e1}, sqMinSemiAx = e[0]>e[1] ? e[1]*e[1] : e[0]*e[0];
  const Real cosang = std::cos(angle), sinang = std::sin(angle);
  const Real bbox[2][2] = {
   { pos[0] -std::max(e0,e1) -safety, pos[0] +std::max(e0,e1) +safety },
   { pos[1] -std::max(e0,e1) -safety, pos[1] +std::max(e0,e1) +safety }
  };

  FillBlocks_Ellipse(const Real _e0, const Real _e1, const Real h,
    const double C[2], Real ang, Real rho): e0(_e0), e1(_e1), safety(2*h),
    pos{(Real)C[0], (Real)C[1]}, angle(ang), rhoS(rho) {}

  inline bool is_touching(const BlockInfo& INFO) const {
    return _is_touching(INFO, bbox, safety);
  }

  void operator()(const BlockInfo& I, ScalarBlock& B, ObstacleBlock& O) const;
};

struct FillBlocks_Plate
{
  //position is not the center of mass, is is the center of the base
  const Real LX, LY, safety, pos[2], angle, angvel, rhoS;
  const Real cosang = std::cos(angle), sinang = std::sin(angle);
  const Real bbox[2][2] = {
    { pos[0] - LX - LY - safety, pos[0] + LX + LY + safety},
    { pos[1] - LX - LY - safety, pos[1] + LX + LY + safety}
  };

  FillBlocks_Plate(Real lx, Real ly, Real h, const double C[2], Real ang,
    Real avel, Real rho): LX(lx), LY(ly), safety(h*2),
    pos{ (Real) C[0], (Real) C[1] }, angle(ang), angvel(avel), rhoS(rho) { }

  inline Real distance(const Real x, const Real y) const {
    const Real X =  x*cosang + y*sinang, Y = -x*sinang + y*cosang;
    return std::min(std::min(X, LX - X), std::min(LY/2 + Y, LY/2 - Y));
  }

  inline bool is_touching(const BlockInfo& INFO) const {
    return _is_touching(INFO, bbox, safety);
  }

  void operator()(const BlockInfo& I, ScalarBlock& B, ObstacleBlock& O) const;
};

struct FillBlocks_VarRhoCylinder
{
  const Real radius, h, rhoTop, rhoBot, angle;
  const Real cosang = std::cos(angle), sinang = std::sin(angle);
  const Real pos[2], bbox[2][2] = {
    { pos[0] - radius - 2*h, pos[0] + radius + 2*h },
    { pos[1] - radius - 2*h, pos[1] + radius + 2*h }
  };

  FillBlocks_VarRhoCylinder(Real R, Real _h, const double C[2], Real rhoT,
    Real rhoB, Real ang) : radius(R), h(_h), rhoTop(rhoT), rhoBot(rhoB),
    angle(ang), pos{ (Real) C[0], (Real) C[1] } {}

  inline Real distanceTocylinder(const Real x, const Real y) const {
      return radius - std::sqrt(x*x+y*y); // pos inside, neg outside
  }

  inline bool is_touching(const BlockInfo& INFO) const {
    return _is_touching(INFO, bbox, 2*h);
  }
  void operator()(const BlockInfo& I, ScalarBlock& B, ObstacleBlock& O) const;
};

struct FillBlocks_VarRhoEllipse
{
  const Real e0, e1, h, pos[2], angle, rhoTop, rhoBot;
  const Real cosang = std::cos(angle), sinang = std::sin(angle);
  const Real e[2] = {e0, e1}, sqMinSemiAx = e[0]>e[1] ? e[1]*e[1] : e[0]*e[0];
  const Real bbox[2][2] = {
    { pos[0] - std::max(e0,e1) - 2*h, pos[0] + std::max(e0,e1) + 2*h },
    { pos[1] - std::max(e0,e1) - 2*h, pos[1] + std::max(e0,e1) + 2*h }
  };

  FillBlocks_VarRhoEllipse(Real _e0, Real _e1, Real _h, const double C[2],
    Real ang, Real rhoT, Real rhoB): e0(_e0), e1(_e1), h(_h),
    pos{(Real)C[0], (Real)C[1]}, angle(ang), rhoTop(rhoT), rhoBot(rhoB) {}

  inline bool is_touching(const BlockInfo& INFO) const {
    return _is_touching(INFO, bbox, 2*h);
  }

  void operator()(const BlockInfo& I, ScalarBlock& B, ObstacleBlock& O) const;
};
