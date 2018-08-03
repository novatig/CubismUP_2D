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

#pragma once

#include "Shape.h"
#define TERM_REW_FAC 50
#define HEIGHT_PENAL 10

class Disk : public Shape
{
  const Real radius;
 public:
  Disk(SimulationData& s, ArgumentParser& p, Real C[2] ) :
  Shape(s,p,C), radius( p("-radius").asDouble(0.1) ) {
    printf("Created a Disk with: R:%f rho:%f\n",radius,rhoS);
  }

  Real getCharLength() const override
  {
    return 2 * radius;
  }

  void outputSettings(ostream &outStream) const override
  {
    outStream << "Disk\n";
    outStream << "radius " << radius << endl;

    Shape::outputSettings(outStream);
  }

  void create(const vector<BlockInfo>& vInfo) override;
};

class HalfDisk : public Shape
{
 protected:
  const Real radius;

 public:
  HalfDisk( SimulationData& s, ArgumentParser& p, Real C[2] ) :
  Shape(s,p,C), radius( p("-radius").asDouble(0.1) ) {
    printf("Created a half Disk with: R:%f rho:%f\n",radius,rhoS);
  }

  Real getCharLength() const override
  {
    return 2 * radius;
  }

  void outputSettings(ostream &outStream) const override
  {
    outStream << "HalfDisk\n";
    outStream << "radius " << radius << endl;

    Shape::outputSettings(outStream);
  }

  void create(const vector<BlockInfo>& vInfo) override;
};

class Ellipse : public Shape
{
 protected:
  const Real semiAxis[2];
  //Characteristic scales:
  const Real majax = std::max(semiAxis[0], semiAxis[1]);
  const Real minax = std::min(semiAxis[0], semiAxis[1]);
  const Real velscale = std::sqrt((rhoS/1-1)*9.81*minax);
  const Real lengthscale = majax, timescale = majax/velscale;
  //const Real torquescale = M_PI/8*pow((a*a-b*b)*velscale,2)/a/b;
  const Real torquescale = M_PI*majax*majax*velscale*velscale;

  Real Torque = 0, old_Torque = 0, old_Dist = 100;
  Real powerOutput = 0, old_powerOutput = 0;

 public:
  Ellipse(SimulationData&s, ArgumentParser&p, Real C[2]) : Shape(s,p,C),
    semiAxis{ (Real) p("-semiAxisX").asDouble(.1),
              (Real) p("-semiAxisY").asDouble(.2) } {
    printf("Created ellipse semiAxis:[%f %f] rhoS:%f a:%f b:%f velscale:%f lengthscale:%f timescale:%f torquescale:%f\n", semiAxis[0], semiAxis[1], rhoS, majax, minax, velscale, lengthscale, timescale, torquescale); fflush(0);
  }

  Real getCharLength() const  override
  {
    return 2 * max(semiAxis[1],semiAxis[0]);
  }

  void outputSettings(ostream &outStream) const override
  {
    outStream << "Ellipse\n";
    outStream << "semiAxisX " << semiAxis[0] << endl;
    outStream << "semiAxisY " << semiAxis[1] << endl;

    Shape::outputSettings(outStream);
  }

  void create(const vector<BlockInfo>& vInfo) override;
};

class DiskVarDensity : public Shape
{
 protected:
  const Real radius;
  const Real rhoTop;
  const Real rhoBot;

 public:
  DiskVarDensity( SimulationData& s, ArgumentParser& p, Real C[2] ) :
  Shape(s,p,C), radius( p("-radius").asDouble(0.1) ),
  rhoTop(p("-rhoTop").asDouble(rhoS) ), rhoBot(p("-rhoBot").asDouble(rhoS) ) {
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
  Real getMinRhoS() const override {
    return std::min(rhoTop, rhoBot);
  }

  void create(const vector<BlockInfo>& vInfo) override;

  void outputSettings(ostream &outStream) const override
  {
    outStream << "DiskVarDensity\n";
    outStream << "radius " << radius << endl;
    outStream << "rhoTop " << rhoTop << endl;
    outStream << "rhoBot " << rhoBot << endl;

    Shape::outputSettings(outStream);
  }
};

class EllipseVarDensity : public Shape
{
  protected:
   const Real semiAxisX;
   const Real semiAxisY;
   const Real rhoTop;
   const Real rhoBot;

  public:
   EllipseVarDensity( SimulationData& s, ArgumentParser& p, Real C[2] ) :
   Shape(s,p,C),
   semiAxisX( p("-semiAxisX").asDouble(0.1) ),
   semiAxisY( p("-semiAxisY").asDouble(0.1) ),
   rhoTop(p("-rhoTop").asDouble(rhoS) ), rhoBot(p("-rhoBot").asDouble(rhoS) ) {
     d_gm[0] = 0;
     // based on weighted average between the centers of mass of half-disks:
     d_gm[1] = -4.*semiAxisY/(3.*M_PI) * (rhoTop-rhoBot)/(rhoTop+rhoBot);

     centerOfMass[0] = center[0] - cos(orientation)*d_gm[0] + sin(orientation)*d_gm[1];
     centerOfMass[1] = center[1] - sin(orientation)*d_gm[0] - cos(orientation)*d_gm[1];
   }

   Real getCharLength() const override {
     return 2 * std::max(semiAxisX, semiAxisY);
   }
   Real getMinRhoS() const override {
     return std::min(rhoTop, rhoBot);
   }

   void create(const vector<BlockInfo>& vInfo) override;

   void outputSettings(ostream &outStream) const override
   {
     outStream << "Ellipse\n";
     outStream << "semiAxisX " << semiAxisX << endl;
     outStream << "semiAxisY " << semiAxisY << endl;
     outStream << "rhoTop " << rhoTop << endl;
     outStream << "rhoBot " << rhoBot << endl;

     Shape::outputSettings(outStream);
   }
};
