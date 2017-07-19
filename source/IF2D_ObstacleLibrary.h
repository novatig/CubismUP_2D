//
//  IF2D_ObstacleLibrary.h
//  IF2D_ROCKS
//
//  Created by Wim van Rees on 04/10/14.
//
//
#ifndef IF2D_ROCKS_IF2D_ObstacleLibrary_h
#define IF2D_ROCKS_IF2D_ObstacleLibrary_h

#include "common.h"
#include "Definitions.h"
#include <map>
#include <limits>
#include <vector>
#include <array>
#include <fstream>

struct FillBlocks_Cylinder
{
    const Real radius, safe_radius, rhoS;
    const double cylinder_position[2];
    Real cylinder_box[2][2];

    void _find_cylinder_box()
    {
        cylinder_box[0][0] = cylinder_position[0] - safe_radius;
        cylinder_box[0][1] = cylinder_position[0] + safe_radius;
        cylinder_box[1][0] = cylinder_position[1] - safe_radius;
        cylinder_box[1][1] = cylinder_position[1] + safe_radius;
    }

    FillBlocks_Cylinder(Real rad, Real h, double pos[2], Real rho):
    radius(rad),safe_radius(rad+4*h),rhoS(rho),cylinder_position{pos[0],pos[1]}
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

    Real getHeavisideFDMH1(const Real x, const Real y, const Real h) const
    {
        const Real dist = distanceTocylinder(x,y);
        if(dist >= +h) return 1;
        if(dist <= -h) return 0;
        assert(std::abs(dist)<=h);

        // compute first primitive of H(x): I(x) = int_0^x H(y) dy
        Real IplusX = distanceTocylinder(x+h,y);
        Real IminuX = distanceTocylinder(x-h,y);
        Real IplusY = distanceTocylinder(x,y+h);
        Real IminuY = distanceTocylinder(x,y-h);


        // set it to zero outside the cylinder
        IplusX = IplusX < 0 ? 0 : IplusX;
        IminuX = IminuX < 0 ? 0 : IminuX;
        IplusY = IplusY < 0 ? 0 : IplusY;
        IminuY = IminuY < 0 ? 0 : IminuY;

        // gradI
        const Real gradIX = 0.5/h * (IplusX - IminuX);
        const Real gradIY = 0.5/h * (IplusY - IminuY);

        // gradU
        const Real gradUX = 0.5/h * ( distanceTocylinder(x+h,y) - distanceTocylinder(x-h,y));
        const Real gradUY = 0.5/h * ( distanceTocylinder(x,y+h) - distanceTocylinder(x,y-h));

        const Real H = (gradIX*gradUX + gradIY*gradUY)/(gradUX*gradUX+gradUY*gradUY);

        return H;
    }

    inline void operator()(const BlockInfo& info, FluidBlock& block, ObstacleBlock * const obstblock) const
    {
      if(_is_touching(info))
      {
        for(int iy=0; iy<FluidBlock::sizeY; iy++)
          for(int ix=0; ix<FluidBlock::sizeX; ix++)
          {
              Real p[2];
              info.pos(p, ix, iy);
							p[0] -= cylinder_position[0];
							p[1] -= cylinder_position[1];
							const Real chi = getHeavisideFDMH1(p[0], p[1], info.h_gridpoint);
							const Real rho = rhoS*chi + block(ix, iy).rho*(1-chi);

							obstblock->chi[iy][ix] = chi;
							obstblock->rho[iy][ix] = rho;
			        block(ix, iy).rho = rho;
          }
      }
    }
};

struct FillBlocks_HalfCylinder
{
  const Real radius, safe_radius;
  const double cylinder_position[2], angle, safety, rhoS;
	const double cosang = std::cos(angle);
	const double sinang = std::sin(angle);
  Real cylinder_box[2][2];

	inline void changeFrame(Real p[2]) const
	{
		const Real x = p[0]-cylinder_position[0];
		const Real y = p[1]-cylinder_position[1];
		p[0] =  x*cosang + y*sinang;
		p[1] = -x*sinang + y*cosang;
	}

  void _find_cylinder_box()
  {
		Real top		= cosang>=0 ?  safe_radius :  safe_radius*abs(sinang);
		top					= max(top,		 safety);

		Real bot		= cosang<=0 ? -safe_radius : -safe_radius*abs(sinang);
		bot 				= min(top,		-safety);

		Real left		= sinang>=0 ? -safe_radius : -safe_radius*abs(cosang);
		left 				= min(left,		-safety);

		Real right	= sinang<=0 ?  safe_radius :  safe_radius*abs(cosang);
		right				= max(right,	 safety);

    cylinder_box[0][0] = cylinder_position[0] + bot;
    cylinder_box[0][1] = cylinder_position[0] + top;
    cylinder_box[1][0] = cylinder_position[1] + left;
    cylinder_box[1][1] = cylinder_position[1] + right;
  }

  FillBlocks_HalfCylinder(Real rad, Real h, double pos[2], Real rho, Real ang):
  radius(rad), safe_radius(rad+4*h), cylinder_position{pos[0],pos[1]}, angle(ang), safety(4*h), rhoS(rho)
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
		if(y<0) return y;
    else return radius - std::sqrt(x*x+y*y); // pos inside, neg outside
  }

  Real getHeavisideFDMH1(const Real x, const Real y, const Real h) const
  {
      const Real dist = distanceTocylinder(x,y);
      if(dist >= +h) return 1;
      if(dist <= -h) return 0;
      assert(std::abs(dist)<=h);

      // compute first primitive of H(x): I(x) = int_0^x H(y) dy
      Real IplusX = distanceTocylinder(x+h,y);
      Real IminuX = distanceTocylinder(x-h,y);
      Real IplusY = distanceTocylinder(x,y+h);
      Real IminuY = distanceTocylinder(x,y-h);


      // set it to zero outside the cylinder
      IplusX = IplusX < 0 ? 0 : IplusX;
      IminuX = IminuX < 0 ? 0 : IminuX;
      IplusY = IplusY < 0 ? 0 : IplusY;
      IminuY = IminuY < 0 ? 0 : IminuY;

      // gradI
      const Real gradIX = 0.5/h * (IplusX - IminuX);
      const Real gradIY = 0.5/h * (IplusY - IminuY);

      // gradU
      const Real gradUX = 0.5/h * ( distanceTocylinder(x+h,y) - distanceTocylinder(x-h,y));
      const Real gradUY = 0.5/h * ( distanceTocylinder(x,y+h) - distanceTocylinder(x,y-h));

      const Real H = (gradIX*gradUX + gradIY*gradUY)/(gradUX*gradUX+gradUY*gradUY);

      return H;
  }

  inline void operator()(const BlockInfo& info, FluidBlock& block, ObstacleBlock * const obstblock) const
  {
    if(_is_touching(info))
    {
      for(int iy=0; iy<FluidBlock::sizeY; iy++)
      for(int ix=0; ix<FluidBlock::sizeX; ix++)
      {
        Real p[2];
        info.pos(p, ix, iy);
				changeFrame(p);
				const Real chi = getHeavisideFDMH1(p[0], p[1], info.h_gridpoint);
				const Real rho = rhoS*chi + block(ix, iy).rho*(1-chi);

        obstblock->chi[iy][ix] = chi;
				obstblock->rho[iy][ix] = rho;
        block(ix, iy).rho = rho;
      }
    }
  }
};

struct FillBlocks_Ellipse
{
	static inline Real DistancePointEllipseSpecial (const Real e[2], const Real y[2], Real x[2])
	{
		if (y[1] > (Real)0)
		{
			if (y[0] > (Real)0)
			{
				// Bisect to compute the root of F(t) for t >= -e1*e1.
				const Real esqr[2] = { e[0]*e[0], e[1]*e[1] };
				const Real ey[2] = { e[0]*y[0], e[1]*y[1] };
				Real t0 = -esqr[1] + ey[1];
				Real t1 = -esqr[1] + sqrt(ey[0]*ey[0] + ey[1]*ey[1]);
				Real t = t0;
				const int imax = 2*std::numeric_limits<Real>::max_exponent;
				for (int i = 0; i < imax; ++i)
				{
					t = ((Real)0.5)*(t0 + t1);
					if ( fabs(t-t0)<2.2e-16 || fabs(t-t1)<2.2e-16 )
					{
						break;
					}

					const Real r[2] = { ey[0]/(t + esqr[0]), ey[1]/(t + esqr[1]) };
					const Real f = r[0]*r[0] + r[1]*r[1] - (Real)1;
					if (f > (Real)0)
					{
						t0 = t;
					}
					else if (f < (Real)0)
					{
						t1 = t;
					}
					else
					{
						break;
					}
				}

				x[0] = esqr[0]*y[0]/(t + esqr[0]);
				x[1] = esqr[1]*y[1]/(t + esqr[1]);
				const Real d[2] = { x[0] - y[0], x[1] - y[1] };
				return sqrt(d[0]*d[0] + d[1]*d[1]);
			}
			else  // y0 == 0
			{
				x[0] = (Real)0;
				x[1] = e[1];
				return fabs(y[1] - e[1]);
			}
		}
		else  // y1 == 0
		{
			const Real denom0 = e[0]*e[0] - e[1]*e[1];
			const Real e0y0 = e[0]*y[0];
			if (e0y0 < denom0)
			{
				// y0 is inside the subinterval.
				const Real x0de0 = e0y0/denom0;
				const Real x0de0sqr = x0de0*x0de0;
				x[0] = e[0]*x0de0;
				x[1] = e[1]*sqrt(fabs((Real)1 - x0de0sqr));
				const Real d0 = x[0] - y[0];
				return sqrt(d0*d0 + x[1]*x[1]);
			}
			else
			{
				// y0 is outside the subinterval.  The closest ellipse point has
				// x1 == 0 and is on the domain-boundary interval (x0/e0)^2 = 1.
				x[0] = e[0];
				x[1] = (Real)0;
				return fabs(y[0] - e[0]);
			}
		}
	}
	//----------------------------------------------------------------------------
	// The ellipse is (x0/e0)^2 + (x1/e1)^2 = 1.  The query point is (y0,y1).
	// The function returns the distance from the query point to the ellipse.
	// It also computes the ellipse point (x0,x1) that is closest to (y0,y1).
	//----------------------------------------------------------------------------

	static Real DistancePointEllipse (const Real e[2], const Real y[2], Real x[2])
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
		for (int i = 0; i < 2; ++i)
		{
			const int j = permute[i];
			locE[i] = e[j];
			locY[i] = y[j];
			if (reflect[j]) locY[i] = -locY[i];
		}

		Real locX[2];
		const Real distance = DistancePointEllipseSpecial(locE, locY, locX);

		// Restore the axis order and reflections.
		for (int i = 0; i < 2; ++i)
		{
			const int j = invpermute[i];
			if (reflect[j]) locX[j] = -locX[j];
			x[i] = locX[j];
		}

		return distance;
	}

  const Real e0,e1,safety;
  const double position[2], angle, rhoS;
	const double cosang = std::cos(angle);
	const double sinang = std::sin(angle);
  Real sphere_box[2][2];

  void _find_cylinder_box()
  {
      const Real maxAxis = std::max(e0,e1);
      sphere_box[0][0] = position[0] - (maxAxis + safety);
      sphere_box[0][1] = position[0] + (maxAxis + safety);
      sphere_box[1][0] = position[1] - (maxAxis + safety);
      sphere_box[1][1] = position[1] + (maxAxis + safety);
  }

  FillBlocks_Ellipse(const Real e0, const Real e1, const Real h, const double pos[2], Real ang, Real rho): e0(e0), e1(e1), safety(4*h), position{pos[0], pos[1]}, angle(ang), rhoS(rho)
  {
		      _find_cylinder_box();
  }

  inline Real mollified_heaviside(const Real x) const
  {
    const Real alpha = M_PI*min(1., max(0., x/safety +.5 ));
    return 0.5+0.5*cos(alpha);
  }

  inline bool _is_touching(const Real min_pos[2], const Real max_pos[2]) const
  {
    Real intersection[2][2] = {
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

  inline void operator()(const BlockInfo& info, FluidBlock& block, ObstacleBlock * const obstblock) const
  {
    const Real Rmatrix[2][2] = {
        {cosang, -sinang},
        {sinang,  cosang}
    };
		const Real e[2] = {e0,e1};
		const Real sqMinSemiAx = e[0]>e[1]  ? e[1]*e[1] : e[0]*e[0];

    if(_is_touching(info))
		{
	    for(int iy=0; iy<FluidBlock::sizeY; iy++)
      for(int ix=0; ix<FluidBlock::sizeX; ix++)
      {
        Real p[2];
        info.pos(p, ix, iy);

        // translate
        p[0] -= position[0];
        p[1] -= position[1];

        // rotate
        const Real t[2] = {
            Rmatrix[0][0]*p[0] + Rmatrix[0][1]*p[1],
            Rmatrix[1][0]*p[0] + Rmatrix[1][1]*p[1]
        };
				const Real sqDist = t[0]*t[0] + t[1]*t[1];

				if (fabs(t[0]) > e[0]+safety*.5 || fabs(t[1]) > e[1]+safety*.5)
				{
					continue; //is outside
				}
				else
				{
					Real rho, chi;
			    if (sqDist < sqMinSemiAx)
					{
						rho = rhoS;
						chi = 1;
					}
					else
					{
						Real xs[2];
		        const Real dist = DistancePointEllipse (e, t, xs);
		        const int sign = sqDist > (xs[0]*xs[0]+xs[1]*xs[1]) ? 1 : -1;
		        chi = mollified_heaviside(sign*dist);
						rho = rhoS*chi + block(ix, iy).rho*(1-chi);
					}

					obstblock->chi[iy][ix] = chi;
					obstblock->rho[iy][ix] = rho;
	        block(ix, iy).rho = rho;
				}
      }
		}
  }
};

struct FillBlocks_Plate
{
	//position is not the center of mass, is is the center of the base
  const Real LX, LY, safety;
  const Real position[2], angle, angvel, rhoS;
	const double cosang = std::cos(angle);
	const double sinang = std::sin(angle);
  Real bbox[2][2];

	inline void toFrame(Real p[2]) const
	{
		const Real x = p[0]-position[0], y = p[1]-position[1];
		p[0] =  x*cosang + y*sinang;
		p[1] = -x*sinang + y*cosang;
	}

  void _find_bbox()
  {
      const double maxreach = LX + LY/2 + safety;
      bbox[0][0] = position[0] - maxreach;
      bbox[0][1] = position[0] + maxreach;
      bbox[1][0] = position[1] - maxreach;
      bbox[1][1] = position[1] + maxreach;
  }

  FillBlocks_Plate(Real lx, Real ly, Real h, const Real pos[2], Real ang, Real avel, Real rho):
  LX(lx), LY(ly), safety(h*4), position{pos[0],pos[1]}, angle(ang), angvel(avel), rhoS(rho)
  {
      _find_bbox();
  }

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
		return min(min(x, LX - x), min(y, LY - y));
  }

	Real getHeavisideFDMH1(const Real x, const Real y, const Real h) const
	{
		const Real dist = distance(x,y);
		if(dist >= +h) return 1;
		if(dist <= -h) return 0;
		assert(std::abs(dist)<=h);

		// compute first primitive of H(x): I(x) = int_0^x H(y) dy
		const Real DplusX = distance(x+h,y);
		const Real DminuX = distance(x-h,y);
		const Real DplusY = distance(x,y+h);
		const Real DminuY = distance(x,y-h);

		// set it to zero outside the cylinder
		const Real IplusX = DplusX < 0 ? 0 : DplusX;
		const Real IminuX = DminuX < 0 ? 0 : DminuX;
		const Real IplusY = DplusY < 0 ? 0 : DplusY;
		const Real IminuY = DminuY < 0 ? 0 : DminuY;

		// gradI
		const Real gradIX = 0.5/h * (IplusX - IminuX);
		const Real gradIY = 0.5/h * (IplusY - IminuY);

		// gradU
		const Real gradUX = 0.5/h * ( IplusX - IminuX);
		const Real gradUY = 0.5/h * ( IplusY - IminuY);

		return (gradIX*gradUX+gradIY*gradUY)/(gradUX*gradUX+gradUY*gradUY+2.2e-16);
	}

  inline void operator()(const BlockInfo& info, FluidBlock& block, ObstacleBlock * const obstblock) const
  {
    if(_is_touching(info))
    for(int iy=0; iy<FluidBlock::sizeY; iy++)
    for(int ix=0; ix<FluidBlock::sizeX; ix++)
    {
        Real p[2];
        info.pos(p, ix, iy);
				toFrame(p);
				const Real chi = getHeavisideFDMH1(p[0], p[1], info.h_gridpoint);
				const Real rho = rhoS*chi + block(ix, iy).rho*(1-chi);

				obstblock->chi[iy][ix] = chi;
				obstblock->rho[iy][ix] = rho;
				obstblock->udef[iy][ix][0] = -p[1]*angvel;
				obstblock->udef[iy][ix][1] =  p[0]*angvel;
				block(ix, iy).rho = rho;
    }
  }
};

#endif
