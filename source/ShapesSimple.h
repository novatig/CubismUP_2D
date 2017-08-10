//
//  Shape.h
//  CubismUP_2D
//
//  Virtual shape class which defines the interface
//  Default simple geometries are also provided and can be used as references
//
//  This class only contains static information (position, orientation,...), no dynamics are included (e.g. velocities,...)
//
//  Created by Christian Conti on 3/6/15.
//  Copyright (c) 2015 ETHZ. All rights reserved.
//

#ifndef CubismUP_2D_ShapeSimple_h
#define CubismUP_2D_ShapeSimple_h

#include "ShapeLibrary.h"
#include "Shape.h"
#define TERM_REW_FAC 50
#define HEIGHT_PENAL 10

class Disk : public Shape
{
 protected:
  Real radius;

 public:
  Disk(Real center[2], Real radius, const Real rhoS) :
  Shape(center, 0, rhoS), radius(radius) { }

  Real getCharLength() const override
  {
    return 2 * radius;
  }

  void outputSettings(ostream &outStream) const
  {
    outStream << "Disk\n";
    outStream << "radius " << radius << endl;

    Shape::outputSettings(outStream);
  }

  void create(const vector<BlockInfo>& vInfo) override
  {
    const Real h =  vInfo[0].h_gridpoint;
    for(auto & entry : obstacleBlocks) delete entry.second;
    obstacleBlocks.clear();

    FillBlocks_Cylinder kernel(radius, h, center, rhoS);
    for(int i=0; i<vInfo.size(); i++) {
      if(kernel._is_touching(vInfo[i])) { //position of sphere + radius + 2*h safety
        assert(obstacleBlocks.find(vInfo[i].blockID) == obstacleBlocks.end());
        obstacleBlocks[vInfo[i].blockID] = new ObstacleBlock;
        obstacleBlocks[vInfo[i].blockID]->clear(); //memset 0
      }
    }

    #pragma omp parallel
    {
      FillBlocks_Cylinder kernel(radius, h, center, rhoS);

      #pragma omp for schedule(dynamic)
      for(int i=0; i<vInfo.size(); i++) {
        const auto pos = obstacleBlocks.find(vInfo[i].blockID);
        if(pos == obstacleBlocks.end()) continue;
        FluidBlock& b = *(FluidBlock*)vInfo[i].ptrBlock;
        kernel(vInfo[i], b, pos->second);
      }
    }

    for (auto & block : obstacleBlocks) block.second->allocate_surface();
  }
};

class DiskVarDensity : public Shape
{
 protected:
  const Real radius;
  const Real rhoTop;
  const Real rhoBot;

 public:
  DiskVarDensity(Real C[2], Real R, Real ang, Real rhoT, Real rhoB) :
  Shape(C, ang, min(rhoT,rhoB)), radius(R), rhoTop(rhoT), rhoBot(rhoB)
  {
    d_gm[0] = 0;
    // based on weighted average between the centers of mass of half-disks:
    d_gm[1] = -4.*radius/(3.*M_PI) * (rhoTop-rhoBot)/(rhoTop+rhoBot);

    centerOfMass[0] = center[0] - cos(orientation)*d_gm[0] + sin(orientation)*d_gm[1];
    centerOfMass[1] = center[1] - sin(orientation)*d_gm[0] - cos(orientation)*d_gm[1];
  }

  Real getCharLength() const  override
  {
    return 2 * radius;
  }

  void create(const vector<BlockInfo>& vInfo) override
  {
    const Real h =  vInfo[0].h_gridpoint;
    for(auto & entry : obstacleBlocks) delete entry.second;
    obstacleBlocks.clear();

    FillBlocks_Cylinder kernel(radius, h, center, rhoS);
    for(int i=0; i<vInfo.size(); i++) {
      if(kernel._is_touching(vInfo[i])) {
        assert(obstacleBlocks.find(vInfo[i].blockID) == obstacleBlocks.end());
        obstacleBlocks[vInfo[i].blockID] = new ObstacleBlock;
        obstacleBlocks[vInfo[i].blockID]->clear(); //memset 0
      }
    }

    #pragma omp parallel
    {
      FillBlocks_Cylinder kernelC(radius, h, center, rhoS);
      //assumption: if touches cylinder, it touches half cylinder:
      FillBlocks_HalfCylinder kernelH(radius, h, center, rhoS, orientation);

      #pragma omp for schedule(dynamic)
      for(int i=0; i<vInfo.size(); i++) {
        BlockInfo info = vInfo[i];
        const auto pos = obstacleBlocks.find(info.blockID);
        if(pos == obstacleBlocks.end()) continue;
        FluidBlock& b = *(FluidBlock*)info.ptrBlock;
        kernelC(info, b, pos->second);
        kernelH(info, b, pos->second);
      }
    }

    for (auto & block : obstacleBlocks) block.second->allocate_surface();
  }

  void outputSettings(ostream &outStream)
  {
    outStream << "DiskVarDensity\n";
    outStream << "radius " << radius << endl;
    outStream << "rhoTop " << rhoTop << endl;
    outStream << "rhoBot " << rhoBot << endl;

    Shape::outputSettings(outStream);
  }
};

class Ellipse : public Shape
{
 protected:
  const Real semiAxis[2];
  //Characteristic scales:
  const Real a=max(semiAxis[0],semiAxis[1]), b=min(semiAxis[0],semiAxis[1]);
  const Real velscale = std::sqrt((rhoS/1.-1)*9.8*b);
  const Real lengthscale = a, timescale = a/velscale;
  //const Real torquescale = M_PI/8*pow((a*a-b*b)*velscale,2)/a/b;
  const Real torquescale = M_PI*a*a*velscale*velscale;

  Real Torque = 0, old_Torque = 0, old_Dist = 100;
  Real powerOutput = 0, old_powerOutput = 0;

 public:
  Ellipse(Real C[2], Real SA[2], Real ang, const Real rho) :
    Shape(C, ang, rho), semiAxis{SA[0],SA[1]}
  {
    printf("Created ellipse semiAxis:[%f %f] rhoS:%f a:%f b:%f velscale:%f lengthscale:%f timescale:%f torquescale:%f\n", semiAxis[0], semiAxis[1], rhoS, a, b, velscale, lengthscale, timescale, torquescale); fflush(0);
  }

  Real getCharLength() const  override
  {
    return 2 * max(semiAxis[1],semiAxis[0]);
  }

  void outputSettings(ostream &outStream) const
  {
    outStream << "Ellipse\n";
    outStream << "semiAxisX " << semiAxis[0] << endl;
    outStream << "semiAxisY " << semiAxis[1] << endl;

    Shape::outputSettings(outStream);
  }

  #ifdef RL_MPI_CLIENT
  void act(Real*const uBody, Real*const vBody, Real*const omegaBody, const Real dt) override
  {
    assert(time_ptr not_eq nullptr);
    assert(communicator not_eq nullptr);
    if(!initialized_time_next_comm || *time_ptr>time_next_comm)
    {
      const Real w = *omegaBody, u = *uBody, v = *vBody;
      const Real cosAng = cos(orientation), sinAng = sin(orientation);
      const Real angle = atan2(sinAng,cosAng);

      //Nondimensionalization:
      const Real xdot = u/velscale, ydot = v/velscale;
      const Real X = labCenterOfMass[0]/a, Y = labCenterOfMass[1]/a;
      const Real U = xdot*cosAng +ydot*sinAng;
      const Real V = ydot*cosAng -xdot*sinAng;
      const Real W = w*timescale;
      const Real T = Torque/torquescale;
      const bool ended = X>125 || X<-10 || Y<=-50;
      const bool landing = std::fabs(angle - .25*M_PI) < 0.1;
      const Real vertDist = std::fabs(Y+50), horzDist = std::fabs(X-100);

      Real reward;
      if (ended)
      {
        info = _AGENT_LASTCOMM;
        reward= (X>125 || X<-10) ? -100 -HEIGHT_PENAL*vertDist
              : (horzDist<1? (landing?2:1) * TERM_REW_FAC : -horzDist) ;
      } else
        reward = (old_Dist-horzDist) -fabs(Torque-old_Torque)/.5;
      //-(powerOutput-old_powerOutput);

      vector<double> state = {U, V, W, X, Y, cosAng, sinAng, T, xdot, ydot}; vector<double> action = {0.};

      printf("Sending (%lu) [%f %f %f %f %f %f %f %f %f %f], %f %f\n",
      state.size(),U,V,W,X,Y,cosAng,sinAng,T,xdot,ydot, Torque,torquescale);

      communicator->sendState(0, info, state, reward);

      if(info == _AGENT_LASTCOMM) abort();
      old_Dist = horzDist;
      old_Torque = Torque;
      old_powerOutput = powerOutput;
      initialized_time_next_comm = true;
      time_next_comm = time_next_comm + 0.5*timescale;
      info = _AGENT_NORMALCOMM;

      communicator->recvAction(action);
         printf("Received %f\n", action[0]);
      Torque = action[0]*torquescale;
    }

    *omegaBody += dt*Torque/J;
    powerOutput += dt*Torque*Torque;
  }
  #endif

  void create(const vector<BlockInfo>& vInfo) override
  {
    const Real h =  vInfo[0].h_gridpoint;
    for(auto & entry : obstacleBlocks) delete entry.second;
    obstacleBlocks.clear();

    FillBlocks_Ellipse kernel(semiAxis[0], semiAxis[1], h, center, orientation, rhoS);
    for(int i=0; i<vInfo.size(); i++) {
      //const auto pos = obstacleBlocks.find(info.blockID);
      if(kernel._is_touching(vInfo[i])) { //position of sphere + radius + 2*h safety
        assert(obstacleBlocks.find(vInfo[i].blockID)==obstacleBlocks.end());
        obstacleBlocks[vInfo[i].blockID] = new ObstacleBlock;
        obstacleBlocks[vInfo[i].blockID]->clear(); //memset 0
      }
    }

    #pragma omp parallel
    {
      FillBlocks_Ellipse kernel(semiAxis[0], semiAxis[1], h, center, orientation, rhoS);

      #pragma omp for schedule(dynamic)
      for(int i=0; i<vInfo.size(); i++) {
        const auto pos = obstacleBlocks.find(vInfo[i].blockID);
        if(pos == obstacleBlocks.end()) continue;
        FluidBlock& b = *(FluidBlock*)vInfo[i].ptrBlock;
        kernel(vInfo[i], b, pos->second);
      }
    }

    const FillBlocks_EllipseFinalize finalize(h, rhoS);
    compute(finalize, vInfo);

    for (auto & block : obstacleBlocks) block.second->allocate_surface();
  }
};

/*
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
*/
#endif
