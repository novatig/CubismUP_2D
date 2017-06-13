//
//  Shape.h
//  CubismUP_2D
//
//	Virtual shape class which defines the interface
//	Default simple geometries are also provided and can be used as references
//
//	This class only contains static information (position, orientation,...), no dynamics are included (e.g. velocities,...)
//
//  Created by Christian Conti on 3/6/15.
//  Copyright (c) 2015 ETHZ. All rights reserved.
//

#ifndef CubismUP_2D_Shape_h
#define CubismUP_2D_Shape_h

class Shape
{
 public:
	Real M = 0;
	Real J = 0;
	Real labCenterOfMass[2] = {0,0};
  Real semiAxis[2] = {0,0};
	// single density
	const Real rhoS;
 protected:
	// general quantities
	Real centerOfMass[2], orientation;
	Real center[2]; // for single density, this corresponds to centerOfMass
	Real d_gm[2] ; // distance of center of geometry to center of mass
	// periodicity - currently unused
    const Real domainSize[2];
    const bool bPeriodic[2];

	// smoothing
	const Real mollChi;
	const Real mollRho; // currently not used - need to change in rho method

	Real smoothHeaviside(Real rR, Real radius, Real eps) const
	{
		if (rR < radius-eps*.5)
			return (Real) 1.;
		else if (rR > radius+eps*.5)
			return (Real) 0.;
		else
			return (Real) ((1.+cos(M_PI*((rR-radius)/eps+.5)))*.5);
	}

public:
	Shape(Real center[2], Real orientation, const Real rhoS, const Real mollChi,
    const Real mollRho, bool bPeriodic[2], Real domainSize[2]) :
		center{center[0],center[1]}, centerOfMass{center[0],center[1]}, d_gm{0,0},
    orientation(orientation), rhoS(rhoS), mollChi(mollChi), mollRho(mollRho),
    domainSize{domainSize[0],domainSize[1]}, bPeriodic{bPeriodic[0],bPeriodic[1]}
	{
		if (bPeriodic[0] || bPeriodic[1])
		{
			cout << "Periodic shapes are currently unsupported\n";
			abort();
		}
	}

	virtual ~Shape() {}

	virtual Real chi(Real p[2], Real h) const = 0;
	virtual Real getCharLength() const = 0;


	void updatePosition(const Real u[2], Real omega, Real dt)
	{
		// update centerOfMass - this is the reference point from which we compute the center
		#ifndef _MOVING_FRAME_
		centerOfMass[0] += dt*u[0];
    centerOfMass[1] += dt*u[1];
		#endif

		labCenterOfMass[0] += dt*u[0];
		labCenterOfMass[1] += dt*u[1];

		orientation += dt*omega;
		orientation = orientation>2*M_PI ? orientation-2*M_PI : (orientation<0 ? orientation+2*M_PI : orientation);

		center[0] = centerOfMass[0] + cos(orientation)*d_gm[0] - sin(orientation)*d_gm[1];
		center[1] = centerOfMass[1] + sin(orientation)*d_gm[0] + cos(orientation)*d_gm[1];
	}

	void setCentroid(Real centroid[2])
	{
		center[0] = centroid[0];
		center[1] = centroid[1];

		centerOfMass[0] = center[0] - cos(orientation)*d_gm[0] + sin(orientation)*d_gm[1];
		centerOfMass[1] = center[1] - sin(orientation)*d_gm[0] - cos(orientation)*d_gm[1];
	}

	void setCenterOfMass(Real com[2])
	{
		centerOfMass[0] = com[0];
		centerOfMass[1] = com[1];

		center[0] = centerOfMass[0] + cos(orientation)*d_gm[0] - sin(orientation)*d_gm[1];
		center[1] = centerOfMass[1] + sin(orientation)*d_gm[0] + cos(orientation)*d_gm[1];
	}

	void getCentroid(Real centroid[2]) const
	{
		centroid[0] = center[0];
		centroid[1] = center[1];
	}

	void getCenterOfMass(Real com[2]) const
	{
		com[0] = centerOfMass[0];
		com[1] = centerOfMass[1];
	}

	void getLabPosition(Real com[2]) const
	{
		com[0] = labCenterOfMass[0];
		com[1] = labCenterOfMass[1];
	}

	Real getOrientation() const
	{
		return orientation;
	}

	virtual inline Real getMinRhoS() const
	{
		return rhoS;
	}

	virtual Real rho(Real p[2], Real h, Real mask) const
	{
		return rhoS*mask + 1.*(1.-mask);
	}

	virtual Real rho(Real p[2], Real h) const
	{
		Real mask = chi(p,h);
		return rho(p,h,mask);
	}

	virtual void outputSettings(ostream &outStream) const
	{
		outStream << "centerX " << center[0] << endl;
		outStream << "centerY " << center[1] << endl;
		outStream << "centerMassX " << centerOfMass[0] << endl;
		outStream << "centerMassY " << centerOfMass[1] << endl;
		outStream << "orientation " << orientation << endl;
		outStream << "rhoS " << rhoS << endl;
		outStream << "mollChi " << mollChi << endl;
		outStream << "mollRho " << mollRho << endl;
	}
};

class Disk : public Shape
{
protected:
	Real radius;

public:
	Disk(Real center[2], Real radius, const Real rhoS, const Real mollChi,
    const Real mollRho, bool bPeriodic[2], Real domainSize[2]) :
		Shape(center, 0, rhoS, mollChi, mollRho, bPeriodic, domainSize),
    radius(radius)
	{
    semiAxis[0] = semiAxis[1] = radius;
	}

	Real chi(Real p[2], Real h) const
	{
		const Real centerPeriodic[2] = {center[0] - floor(center[0]/domainSize[0]) * bPeriodic[0],
										center[1] - floor(center[1]/domainSize[1]) * bPeriodic[1]};

		const Real d[2] = { abs(p[0]-centerPeriodic[0]), abs(p[1]-centerPeriodic[1]) };
		const Real dist = sqrt(d[0]*d[0]+d[1]*d[1]);

		return smoothHeaviside(dist, radius, mollChi*sqrt(2)*h);
	}

	Real getCharLength() const
	{
		return 2 * radius;
	}

	void outputSettings(ostream &outStream) const
	{
		outStream << "Disk\n";
		outStream << "radius " << radius << endl;

		Shape::outputSettings(outStream);
	}
};

class DiskVarDensity : public Shape
{
 protected:
	Real radius;
	Real rhoS1, rhoS2;

 public:
	DiskVarDensity(Real center[2], const Real radius, const Real orientation, const Real rhoS1, const Real rhoS2, const Real mollChi, const Real mollRho, bool bPeriodic[2], Real domainSize[2]) : Shape(center, orientation, min(rhoS1,rhoS2), mollChi, mollRho, bPeriodic, domainSize), radius(radius), rhoS1(rhoS1), rhoS2(rhoS2)
	{
		d_gm[0] = 0;
		d_gm[1] = -4.*radius/(3.*M_PI) * (rhoS1-rhoS2)/(rhoS1+rhoS2); // based on weighted average between the centers of mass of half-disks

		centerOfMass[0] = center[0] - cos(orientation)*d_gm[0] + sin(orientation)*d_gm[1];
		centerOfMass[1] = center[1] - sin(orientation)*d_gm[0] - cos(orientation)*d_gm[1];
	}

	Real chi(Real p[2], Real h) const
	{
		// this part remains as for the constant density disk
		const Real centerPeriodic[2] = {center[0] - floor(center[0]/domainSize[0]) * bPeriodic[0],
										center[1] - floor(center[1]/domainSize[1]) * bPeriodic[1]};

		const Real d[2] = { abs(p[0]-centerPeriodic[0]), abs(p[1]-centerPeriodic[1]) };
		const Real dist = sqrt(d[0]*d[0]+d[1]*d[1]);

		return smoothHeaviside(dist, radius, mollChi*sqrt(2)*h);
	}

	Real rho(Real p[2], Real h, Real mask) const
	{
		// not handling periodicity

		Real r = 0;
		if (orientation == 0 || orientation == 2*M_PI)
			r = smoothHeaviside(p[1],center[1], mollRho*sqrt(2)*h);
		else if (orientation == M_PI)
			r = smoothHeaviside(center[1],p[1], mollRho*sqrt(2)*h);
		else if (orientation == M_PI_2)
			r = smoothHeaviside(center[0],p[0], mollRho*sqrt(2)*h);
		else if (orientation == 3*M_PI_2)
			r = smoothHeaviside(p[0],center[0], mollRho*sqrt(2)*h);
		else
		{
			const Real tantheta = tan(orientation);
			r = smoothHeaviside(p[1], tantheta*p[0]+center[1]-tantheta*center[0], mollRho*sqrt(2)*h);
			r = (orientation>M_PI_2 && orientation<3*M_PI_2) ? 1-r : r;
		}

		return ((rhoS2-rhoS1)*r+rhoS1)*mask + 1.*(1.-mask);
	}

	Real rho(Real p[2], Real h) const
	{
		Real mask = chi(p,h);
		return rho(p,h,mask);
	}

	Real getCharLength() const
	{
		return 2 * radius;
	}

	void outputSettings(ostream &outStream)
	{
		outStream << "DiskVarDensity\n";
		outStream << "radius " << radius << endl;
		outStream << "rhoS1 " << rhoS1 << endl;
		outStream << "rhoS2 " << rhoS2 << endl;

		Shape::outputSettings(outStream);
	}
};

class Ellipse : public Shape
{
 protected:
	// these quantities are defined in the local coordinates of the ellipse

	// code from http://www.geometrictools.com/
	//----------------------------------------------------------------------------
	// The ellipse is (x0/semiAxis0)^2 + (x1/semiAxis1)^2 = 1.  The query point is (y0,y1).
	// The function returns the distance from the query point to the ellipse.
	// It also computes the ellipse point (x0,x1) that is closest to (y0,y1).
	//----------------------------------------------------------------------------
	inline Real DistancePointEllipseSpecial (const Real e[2], const Real y[2], Real x[2]) const
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
					if (t == t0 || t == t1)
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

	inline Real DistancePointEllipse(const Real y[2], Real x[2]) const
	{
		// Determine reflections for y to the first quadrant.
		bool reflect[2];
		for (int i = 0; i < 2; ++i)
		{
			reflect[i] = (y[i] < (Real)0);
		}

		// Determine the axis order for decreasing extents.
		int permute[2];
		if (semiAxis[0] < semiAxis[1])
		{
			permute[0] = 1;  permute[1] = 0;
		}
		else
		{
			permute[0] = 0;  permute[1] = 1;
		}

		int invpermute[2];
		for (int i = 0; i < 2; ++i)
		{
			invpermute[permute[i]] = i;
		}

		Real locE[2], locY[2];
		for (int i = 0; i < 2; ++i)
		{
			const int j = permute[i];
			locE[i] = semiAxis[j];
			locY[i] = y[j];
			if (reflect[j])
			{
				locY[i] = -locY[i];
			}
		}

		Real locX[2];
		const Real distance = DistancePointEllipseSpecial(locE, locY, locX);

		// Restore the axis order and reflections.
		for (int i = 0; i < 2; ++i)
		{
			const int j = invpermute[i];
			if (reflect[j])
			{
				locX[j] = -locX[j];
			}
			x[i] = locX[j];
		}

		return distance;
	}

 public:
	Ellipse(Real center[2], Real sA[2], Real orientation, const Real rhoS,
    const Real mollChi, const Real mollRho, bool bPeriodic[2], Real domainSize[2]) :
    Shape(center, orientation, rhoS, mollChi, mollRho, bPeriodic, domainSize)
    {
      semiAxis[0] = sA[0]; 
      semiAxis[1] = sA[1];
    }

	Real chi(Real p[2], Real h) const
  {
		Real x[2] = {0,0};
		const Real pShift[2] = {p[0]-center[0],p[1]-center[1]};
    const Real eps = mollChi*sqrt(2)*h;

		const Real rotatedP[2] = {
        cos(orientation)*pShift[1] - sin(orientation)*pShift[0],
				sin(orientation)*pShift[1] + cos(orientation)*pShift[0]
    };

    if (std::fabs(rotatedP[0]) > semiAxis[0] + eps*.5 ) return 0;
    if (std::fabs(rotatedP[1]) > semiAxis[1] + eps*.5 ) return 0;
    const Real sqDist = rotatedP[0]*rotatedP[0] + rotatedP[1]*rotatedP[1];
    const Real sqMinSemiAx = semiAxis[0]>semiAxis[1]  ? semiAxis[1]*semiAxis[1]
                                                      : semiAxis[0]*semiAxis[0];
    if (sqDist < sqMinSemiAx)  return 1;

		const Real dist = DistancePointEllipse(rotatedP, x);
		const int sign = ( sqDist > (x[0]*x[0]+x[1]*x[1]) ) ? 1 : -1;

    if (sign*dist < -eps*.5) return 1;
    if (sign*dist >  eps*.5) return 0;
    return (1.+cos(M_PI*(sign*dist/eps+.5)))*.5;
	}

	Real getCharLength() const
	{
		return 2 * semiAxis[1];
	}

	void outputSettings(ostream &outStream) const
	{
		outStream << "Ellipse\n";
		outStream << "semiAxisX " << semiAxis[0] << endl;
		outStream << "semiAxisY " << semiAxis[1] << endl;

		Shape::outputSettings(outStream);
	}
};


class EllipseVarDensity : public Shape
{
 protected:
	// these quantities are defined in the local coordinates of the ellipse
	Real semiAxis[2];
	Real rhoS1, rhoS2;

	// code from http://www.geometrictools.com/
	//----------------------------------------------------------------------------
	// The ellipse is (x0/semiAxis0)^2 + (x1/semiAxis1)^2 = 1.  The query point is (y0,y1).
	// The function returns the distance from the query point to the ellipse.
	// It also computes the ellipse point (x0,x1) that is closest to (y0,y1).
	//----------------------------------------------------------------------------
	Real DistancePointEllipseSpecial (const Real e[2], const Real y[2], Real x[2]) const
	{
		Real distance = (Real)0;
		if (y[1] > (Real)0)
		{
			if (y[0] > (Real)0)
			{
				// Bisect to compute the root of F(t) for t >= -e1*e1.
				Real esqr[2] = { e[0]*e[0], e[1]*e[1] };
				Real ey[2] = { e[0]*y[0], e[1]*y[1] };
				Real t0 = -esqr[1] + ey[1];
				Real t1 = -esqr[1] + sqrt(ey[0]*ey[0] + ey[1]*ey[1]);
				Real t = t0;
				const int imax = 2*std::numeric_limits<Real>::max_exponent;
				for (int i = 0; i < imax; ++i)
				{
					t = ((Real)0.5)*(t0 + t1);
					if (t == t0 || t == t1)
					{
						break;
					}

					Real r[2] = { ey[0]/(t + esqr[0]), ey[1]/(t + esqr[1]) };
					Real f = r[0]*r[0] + r[1]*r[1] - (Real)1;
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
				Real d[2] = { x[0] - y[0], x[1] - y[1] };
				distance = sqrt(d[0]*d[0] + d[1]*d[1]);
			}
			else  // y0 == 0
			{
				x[0] = (Real)0;
				x[1] = e[1];
				distance = fabs(y[1] - e[1]);
			}
		}
		else  // y1 == 0
		{
			Real denom0 = e[0]*e[0] - e[1]*e[1];
			Real e0y0 = e[0]*y[0];
			if (e0y0 < denom0)
			{
				// y0 is inside the subinterval.
				Real x0de0 = e0y0/denom0;
				Real x0de0sqr = x0de0*x0de0;
				x[0] = e[0]*x0de0;
				x[1] = e[1]*sqrt(fabs((Real)1 - x0de0sqr));
				Real d0 = x[0] - y[0];
				distance = sqrt(d0*d0 + x[1]*x[1]);
			}
			else
			{
				// y0 is outside the subinterval.  The closest ellipse point has
				// x1 == 0 and is on the domain-boundary interval (x0/e0)^2 = 1.
				x[0] = e[0];
				x[1] = (Real)0;
				distance = fabs(y[0] - e[0]);
			}
		}
		return distance;
	}

	Real DistancePointEllipse(const Real y[2], Real x[2]) const
	{
		// Determine reflections for y to the first quadrant.
		bool reflect[2];
		int i, j;
		for (i = 0; i < 2; ++i)
		{
			reflect[i] = (y[i] < (Real)0);
		}

		// Determine the axis order for decreasing extents.
		int permute[2];
		if (semiAxis[0] < semiAxis[1])
		{
			permute[0] = 1;  permute[1] = 0;
		}
		else
		{
			permute[0] = 0;  permute[1] = 1;
		}

		int invpermute[2];
		for (i = 0; i < 2; ++i)
		{
			invpermute[permute[i]] = i;
		}

		Real locE[2], locY[2];
		for (i = 0; i < 2; ++i)
		{
			j = permute[i];
			locE[i] = semiAxis[j];
			locY[i] = y[j];
			if (reflect[j])
			{
				locY[i] = -locY[i];
			}
		}

		Real locX[2];
		Real distance = DistancePointEllipseSpecial(locE, locY, locX);

		// Restore the axis order and reflections.
		for (i = 0; i < 2; ++i)
		{
			j = invpermute[i];
			if (reflect[j])
			{
				locX[j] = -locX[j];
			}
			x[i] = locX[j];
		}

		return distance;
	}

 public:
	EllipseVarDensity(Real center[2], Real semiAxis[2], Real orientation, const Real rhoS1, const Real rhoS2, const Real mollChi, const Real mollRho, bool bPeriodic[2], Real domainSize[2]) : Shape(center, orientation, min(rhoS1,rhoS2), mollChi, mollRho, bPeriodic, domainSize), semiAxis{semiAxis[0],semiAxis[1]}, rhoS1(rhoS1), rhoS2(rhoS2)
	{
		d_gm[0] = 0;
		d_gm[1] = -4.*semiAxis[0]/(3.*M_PI) * (rhoS1-rhoS2)/(rhoS1+rhoS2); // based on weighted average between the centers of mass of half-disks

		centerOfMass[0] = center[0] - cos(orientation)*d_gm[0] + sin(orientation)*d_gm[1];
		centerOfMass[1] = center[1] - sin(orientation)*d_gm[0] - cos(orientation)*d_gm[1];
	}

	Real chi(Real p[2], Real h) const
	{
		const Real centerPeriodic[2] = {center[0] - floor(center[0]/domainSize[0]) * bPeriodic[0],
										center[1] - floor(center[1]/domainSize[1]) * bPeriodic[1]};
		Real x[2] = {0,0};
		const Real pShift[2] = {p[0]-centerPeriodic[0],p[1]-centerPeriodic[1]};

		const Real rotatedP[2] = { cos(orientation)*pShift[1] - sin(orientation)*pShift[0],
								   sin(orientation)*pShift[1] + cos(orientation)*pShift[0] };
		const Real dist = DistancePointEllipse(rotatedP, x);
		const int sign = ( (rotatedP[0]*rotatedP[0]+rotatedP[1]*rotatedP[1]) > (x[0]*x[0]+x[1]*x[1]) ) ? 1 : -1;

		return smoothHeaviside(sign*dist,0,mollChi*sqrt(2)*h);
	}

	Real rho(Real p[2], Real h, Real mask) const
	{
		// not handling periodicity

		Real r = 0;
		if (orientation == 0 || orientation == 2*M_PI)
			r = smoothHeaviside(p[1],center[1], mollRho*sqrt(2)*h);
		else if (orientation == M_PI)
			r = smoothHeaviside(center[1],p[1], mollRho*sqrt(2)*h);
		else if (orientation == M_PI_2)
			r = smoothHeaviside(center[0],p[0], mollRho*sqrt(2)*h);
		else if (orientation == 3*M_PI_2)
			r = smoothHeaviside(p[0],center[0], mollRho*sqrt(2)*h);
		else
		{
			const Real tantheta = tan(orientation);
			r = smoothHeaviside(p[1], tantheta*p[0]+center[1]-tantheta*center[0], mollRho*sqrt(2)*h);
			r = (orientation>M_PI_2 && orientation<3*M_PI_2) ? 1-r : r;
		}

		return ((rhoS2-rhoS1)*r+rhoS1)*mask + 1.*(1.-mask);
	}

	Real rho(Real p[2], Real h) const
	{
		Real mask = chi(p,h);
		return rho(p,h,mask);
	}

	Real getCharLength() const
	{
		return 2 * semiAxis[1];
	}

	void outputSettings(ostream &outStream) const
	{
		outStream << "Ellipse\n";
		outStream << "semiAxisX " << semiAxis[0] << endl;
		outStream << "semiAxisY " << semiAxis[1] << endl;
		outStream << "rhoS1 " << rhoS1 << endl;
		outStream << "rhoS2 " << rhoS2 << endl;

		Shape::outputSettings(outStream);
	}
};

#endif
