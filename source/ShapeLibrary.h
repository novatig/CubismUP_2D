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

#include "common.h"
#include "Definitions.h"
#include "ObstacleBlock.h"
#include <map>
#include <limits>
#include <vector>
#include <array>
#include <fstream>

struct FillBlocks_Cylinder
{
  const Real radius, safe_radius, rhoS;
  const Real cylinder_position[2];
  Real cylinder_box[2][2];

  void _find_cylinder_box()
  {
    cylinder_box[0][0] = cylinder_position[0] - safe_radius;
    cylinder_box[0][1] = cylinder_position[0] + safe_radius;
    cylinder_box[1][0] = cylinder_position[1] - safe_radius;
    cylinder_box[1][1] = cylinder_position[1] + safe_radius;
  }

  FillBlocks_Cylinder(Real rad, Real h, Real pos[2], Real rho):
  radius(rad),safe_radius(rad+2*h),rhoS(rho),cylinder_position{pos[0],pos[1]}
  {
      _find_cylinder_box();
  }

  bool _is_touching(const Real min_pos[2], const Real max_pos[2]) const
  {
    Real intersection[2][2] =
    {
        max(min_pos[0], cylinder_box[0][0]), min(max_pos[0], cylinder_box[0][1]),
        max(min_pos[1], cylinder_box[1][0]), min(max_pos[1], cylinder_box[1][1])
    };

    return
    intersection[0][1]-intersection[0][0]>0 &&
    intersection[1][1]-intersection[1][0]>0;
  }

  bool _is_touching(const BlockInfo& info, const int buffer_dx = 0) const
  {
    Real min_pos[2], max_pos[2];

    info.pos(min_pos, 0,0);
    info.pos(max_pos, FluidBlock::sizeX-1, FluidBlock::sizeY-1);
    for(int i=0;i<2;++i)
    {
        min_pos[i]-=buffer_dx*info.h_gridpoint;
        max_pos[i]+=buffer_dx*info.h_gridpoint;
    }
    return _is_touching(min_pos,max_pos);
  }

  inline Real distanceTocylinder(const Real x, const Real y) const
  {
      return radius - std::sqrt(x*x+y*y); // pos inside, neg outside
  }

  void operator()(const BlockInfo& info, FluidBlock& block, ObstacleBlock * const obstblock) const;
};

struct FillBlocks_HalfCylinder
{
  const Real radius, safe_radius;
  const Real cylinder_position[2], angle, safety, rhoS;
  const Real cosang = std::cos(angle);
  const Real sinang = std::sin(angle);
  const Real cylinder_box[2][2] = {
   { cylinder_position[0] -safe_radius, cylinder_position[0] +safe_radius },
   { cylinder_position[1] -safe_radius, cylinder_position[1] +safe_radius }
  };

  FillBlocks_HalfCylinder(Real rad, Real h, Real pos[2], Real rho, Real ang):
  radius(rad), safe_radius(rad+2*h), cylinder_position{pos[0],pos[1]}, angle(ang), safety(2*h), rhoS(rho) { }

  bool _is_touching(const Real min_pos[2], const Real max_pos[2]) const
  {
    Real intersection[2][2] = {
      max(min_pos[0], cylinder_box[0][0]), min(max_pos[0], cylinder_box[0][1]),
      max(min_pos[1], cylinder_box[1][0]), min(max_pos[1], cylinder_box[1][1])
    };

    return
    intersection[0][1]-intersection[0][0]>0 &&
    intersection[1][1]-intersection[1][0]>0;
  }

  bool _is_touching(const BlockInfo& info, const int buffer_dx = 0) const
  {
    Real min_pos[2], max_pos[2];
    info.pos(min_pos, 0,0);
    info.pos(max_pos, FluidBlock::sizeX-1, FluidBlock::sizeY-1);
    for(int i=0;i<2;++i) {
        min_pos[i]-=buffer_dx*info.h_gridpoint;
        max_pos[i]+=buffer_dx*info.h_gridpoint;
    }
    return _is_touching(min_pos,max_pos);
  }

  inline Real distanceTocylinder(const Real x, const Real y) const {
    const Real X =   x*cosang + y*sinang;
    if(X>0) return -X;
    //const Real Y = - x*sinang + y*cosang; /// For default orientation
    //if(Y>0) return -Y;                    /// pointing downwards.
    else return radius - std::sqrt(x*x+y*y); // (pos inside, neg outside)
  }

  void operator()(const BlockInfo& info, FluidBlock& block, ObstacleBlock * const obstblock) const;
};

struct FillBlocks_Ellipse
{
  static inline Real DistancePointEllipseSpecial(const Real e[2], const Real y[2], Real x[2]);

  static Real DistancePointEllipse(const Real e[2], const Real y[2], Real x[2]);

  const Real e0,e1,safety;
  const Real position[2], angle, rhoS;
  const Real cosang = std::cos(angle);
  const Real sinang = std::sin(angle);
  const Real sphere_box[2][2] = {
   { position[0] -std::max(e0,e1)-safety, position[0] +std::max(e0,e1)+safety },
   { position[1] -std::max(e0,e1)-safety, position[1] +std::max(e0,e1)+safety }
  };

  FillBlocks_Ellipse(const Real _e0, const Real _e1, const Real h,
    const Real pos[2], Real ang, Real rho): e0(_e0), e1(_e1), safety(2*h),
    position{pos[0], pos[1]}, angle(ang), rhoS(rho) {}

  inline bool _is_touching(const Real min_pos[2], const Real max_pos[2]) const
  {
    const Real intersection[2][2] = {
        max(min_pos[0], sphere_box[0][0]), min(max_pos[0], sphere_box[0][1]),
        max(min_pos[1], sphere_box[1][0]), min(max_pos[1], sphere_box[1][1])
    };
    return
    intersection[0][1]-intersection[0][0]>0 &&
    intersection[1][1]-intersection[1][0]>0;
  }

  bool _is_touching(const BlockInfo& info, const int buffer_dx=0) const
  {
    Real min_pos[2], max_pos[2];

    info.pos(min_pos, 0,0);
    info.pos(max_pos, FluidBlock::sizeX-1, FluidBlock::sizeY-1);
    for(int i=0;i<2;++i) {
        min_pos[i] -= buffer_dx*info.h_gridpoint;
        max_pos[i] += buffer_dx*info.h_gridpoint;
    }
    return _is_touching(min_pos,max_pos);
  }

  void operator()(const BlockInfo& info, FluidBlock& block, ObstacleBlock * const obstblock) const;
};

struct FillBlocks_EllipseFinalize
{
  const Real h, rhoS;
  const int stencil_start[3] = {-1, -1, 0}, stencil_end[3] = {2, 2, 1};
  StencilInfo stencil;

  FillBlocks_EllipseFinalize(const Real _h, Real rho): h(_h), rhoS(rho)
  {
    stencil = StencilInfo(-1,-1,0, 2,2,1, false, 1, 5);
  }

  void operator()(Lab& l, const BlockInfo&i, FluidBlock&b, ObstacleBlock*const o) const;
};

struct FillBlocks_Plate
{
  //position is not the center of mass, is is the center of the base
  const Real LX, LY, safety;
  const Real position[2], angle, angvel, rhoS;
  const Real cosang = std::cos(angle);
  const Real sinang = std::sin(angle);
  const Real bbox[2][2] = {
    {position[0] -LX-LY-safety, position[0] +LX+LY+safety},
    {position[1] -LX-LY-safety, position[1] +LX+LY+safety}
  };

  FillBlocks_Plate(Real lx, Real ly, Real h, const Real pos[2], Real ang, Real avel, Real rho):
  LX(lx), LY(ly), safety(h*2), position{pos[0],pos[1]}, angle(ang), angvel(avel), rhoS(rho) { }

  bool _is_touching(const BlockInfo& info) const
  {
    Real min_pos[2], max_pos[2];
    info.pos(min_pos, 0,0);
    info.pos(max_pos, FluidBlock::sizeX-1, FluidBlock::sizeY-1);

    Real intersection[2][2] = {
        max(min_pos[0], bbox[0][0]), min(max_pos[0], bbox[0][1]),
        max(min_pos[1], bbox[1][0]), min(max_pos[1], bbox[1][1]),
    };

    return
    intersection[0][1]-intersection[0][0]>0 &&
    intersection[1][1]-intersection[1][0]>0 ;
  }

  inline Real distance(const Real x, const Real y) const
  {  // pos inside, neg outside
    const Real X =  x*cosang + y*sinang;
    const Real Y = -x*sinang + y*cosang;
    return min(min(X, LX - X), min(LY/2 + Y, LY/2 - Y));
  }

  void operator()(const BlockInfo& info, FluidBlock& block, ObstacleBlock * const obstblock) const;
};

struct FillBlocks_VarRhoCylinder
{
  const Real radius, safe_radius, rhoTop, rhoBot, angle;
  const Real cosang = std::cos(angle);
  const Real sinang = std::sin(angle);
  const Real cylinder_position[2];
  const Real cylinder_box[2][2] = {
    {cylinder_position[0]-safe_radius, cylinder_position[0]+safe_radius},
    {cylinder_position[1]-safe_radius, cylinder_position[1]+safe_radius}
  };

  inline Real mollified_heaviside(const Real x) const
  {
    const Real alpha = M_PI*min(1., max(0., .5*x + .5));
    return 0.5+0.5*std::cos(alpha);
  }

  FillBlocks_VarRhoCylinder(Real rad, Real h, Real pos[2], Real rhoT, Real rhoB, Real ang) : radius(rad), safe_radius(rad+2*h), rhoTop(rhoT), rhoBot(rhoB), angle(ang), cylinder_position{pos[0],pos[1]} {}

  bool _is_touching(const Real min_pos[2], const Real max_pos[2]) const
  {
    Real intersection[2][2] = {
      max(min_pos[0], cylinder_box[0][0]), min(max_pos[0], cylinder_box[0][1]),
      max(min_pos[1], cylinder_box[1][0]), min(max_pos[1], cylinder_box[1][1])
    };

    return
    intersection[0][1]-intersection[0][0]>0 &&
    intersection[1][1]-intersection[1][0]>0;
  }

  inline bool _is_touching(const BlockInfo& info, const int buffer_dx=0) const
  {
    Real min_pos[2], max_pos[2];

    info.pos(min_pos, 0,0);
    info.pos(max_pos, FluidBlock::sizeX-1, FluidBlock::sizeY-1);
    for(int i=0;i<2;++i) {
      min_pos[i]-=buffer_dx*info.h_gridpoint;
      max_pos[i]+=buffer_dx*info.h_gridpoint;
    }
    return _is_touching(min_pos,max_pos);
  }

  inline Real distanceTocylinder(const Real x, const Real y) const
  {
      return radius - std::sqrt(x*x+y*y); // pos inside, neg outside
  }

  void operator()(const BlockInfo& info, FluidBlock& block, ObstacleBlock * const obstblock) const;
};

struct FillBlocks_VarRhoEllipseFinalize
{
  const Real h, rhoTop, rhoBot, angle, position[2];
  const Real cosang = std::cos(angle);
  const Real sinang = std::sin(angle);
  const int stencil_start[3] = {-1, -1, 0}, stencil_end[3] = {2, 2, 1};

  inline Real mollified_heaviside(const Real x) const
  {
    const Real alpha = M_PI*min(1., max(0., .5*x + .5));
    return 0.5+0.5*std::cos(alpha);
  }

  FillBlocks_VarRhoEllipseFinalize(Real _h, Real C[2], Real rhoT, Real rhoB, Real ang) : h(_h), rhoTop(rhoT), rhoBot(rhoB), angle(ang), position{C[0],C[1]} {}

  void operator()(Lab& l, const BlockInfo&i, FluidBlock&b, ObstacleBlock*const o) const;
};
