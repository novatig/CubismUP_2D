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
#include "OperatorComputeForces.h"

class Shape
{
 protected: // data fields
  SimulationData& sim;
  // general quantities
  const Real origC[2], origAng;
  Real center[2]; // for single density, this corresponds to centerOfMass
  Real centerOfMass[2];
  Real d_gm[2]; // distance of center of geometry to center of mass
  Real labCenterOfMass[2] = {0,0};
  Real orientation;
  Real M = 0;
  Real V = 0;
  Real J = 0;
  Real u = 0; // in lab frame, not sim frame
  Real v = 0; // in lab frame, not sim frame
  Real omega = 0;
  Real computedu = 0;
  Real computedv = 0;
  Real computedo = 0;
  std::map<int,ObstacleBlock*> obstacleBlocks;

 public: // data fields
  Real dragP[2] = {0,0};
  Real dragV = 0;
  const Real rhoS;
  const bool bForced;
  const bool bFixed;
  const bool bForcedx;
  const bool bForcedy;
  const bool bBlockang;
  const bool bFixedx;
  const bool bFixedy;
  const Real forcedu;
  const Real forcedv;

  virtual void resetAll() {
             center[0] = origC[0];
             center[1] = origC[1];
       centerOfMass[0] = origC[0];
       centerOfMass[1] = origC[1];
    labCenterOfMass[0] = 0;
    labCenterOfMass[1] = 0;
    orientation = origAng;
    M = 0;
    V = 0;
    J = 0;
    u = 0;
    v = 0;
    omega = 0;
    computedu = 0;
    computedv = 0;
    computedo = 0;
    d_gm[0] = 0;
    d_gm[1] = 0;
    for(auto & entry : obstacleBlocks) delete entry.second;
    obstacleBlocks.clear();
  }

 protected:

  Real smoothHeaviside(Real rR, Real radius, Real eps) const
  {
    if (rR < radius-eps*.5) return (Real) 1.;
    else if (rR > radius+eps*.5) return (Real) 0.;
    else return (Real) ((1.+cos(M_PI*((rR-radius)/eps+.5)))*.5);
  }

  inline void rotate(Real p[2]) const
  {
    const Real x = p[0], y = p[1];
    p[0] =  x*std::cos(orientation) + y*std::sin(orientation);
    p[1] = -x*std::sin(orientation) + y*std::cos(orientation);
  }

 public:
  Shape( SimulationData& s, ArgumentParser& p, Real C[2] ) :
  sim(s), origC{C[0],C[1]}, origAng( p("-angle").asDouble(0)*M_PI/180 ),
  center{C[0],C[1]}, centerOfMass{C[0],C[1]},
  d_gm{0,0}, orientation(origAng),
  rhoS( p("-rhoS").asDouble(1) ),
  bForced( p("-bForced").asBool(false) ),
  bFixed( p("-bFixed").asBool(false) ),
  bForcedx(p("-bForcedx").asBool(bForced)),
  bForcedy(p("-bForcedy").asBool(bForced)),
  bBlockang( p("-bBlockAng").asBool(bForcedx || bForcedy) ),
  bFixedx(p("-bFixedx" ).asBool(bFixed) ),
  bFixedy(p("-bFixedy" ).asBool(bFixed) ),
  forcedu( - p("-xvel").asDouble(0) ),
  forcedv( - p("-yvel").asDouble(0) )
  {  }

  virtual ~Shape() {
    for(auto & entry : obstacleBlocks) delete entry.second;
    obstacleBlocks.clear();
  }

  virtual Real getCharLength() const = 0;
  virtual void create(const vector<BlockInfo>& vInfo) = 0;

  virtual void updatePosition(double dt)
  {
    // update centerOfMass - this is the reference point from which we compute the center
    // Remember, uinf is -ubox, therefore we sum it to u body to get
    // velocity of shapre relative to the sim box
    centerOfMass[0] += dt*( u + sim.uinfx );
    centerOfMass[1] += dt*( v + sim.uinfy );
    labCenterOfMass[0] += dt*u;
    labCenterOfMass[1] += dt*v;

    orientation += dt*omega;
    orientation = orientation>2*M_PI ? orientation-2*M_PI : (orientation<0 ? orientation+2*M_PI : orientation);

    center[0] = centerOfMass[0] + std::cos(orientation)*d_gm[0] - std::sin(orientation)*d_gm[1];
    center[1] = centerOfMass[1] + std::sin(orientation)*d_gm[0] + std::cos(orientation)*d_gm[1];
  }

  void setCentroid(Real centroid[2])
  {
    center[0] = centroid[0];
    center[1] = centroid[1];

    centerOfMass[0] = center[0] - std::cos(orientation)*d_gm[0] + std::sin(orientation)*d_gm[1];
    centerOfMass[1] = center[1] - std::sin(orientation)*d_gm[0] - std::cos(orientation)*d_gm[1];
  }

  void setCenterOfMass(Real com[2])
  {
    centerOfMass[0] = com[0];
    centerOfMass[1] = com[1];

    center[0] = centerOfMass[0] + std::cos(orientation)*d_gm[0] - std::sin(orientation)*d_gm[1];
    center[1] = centerOfMass[1] + std::sin(orientation)*d_gm[0] + std::cos(orientation)*d_gm[1];
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

  Real getU() const { return u; }
  Real getV() const { return v; }
  Real getW() const { return omega; }

  Real getOrientation() const
  {
    return orientation;
  }
  void setOrientation(const Real angle)
  {
    orientation = angle;
  }

  virtual inline Real getMinRhoS() const
  {
    return rhoS;
  }

  virtual void outputSettings(ostream &outStream) const
  {
    outStream << "centerX " << center[0] << endl;
    outStream << "centerY " << center[1] << endl;
    outStream << "centerMassX " << centerOfMass[0] << endl;
    outStream << "centerMassY " << centerOfMass[1] << endl;
    outStream << "orientation " << orientation << endl;
    outStream << "rhoS " << rhoS << endl;
  }

  struct Integrals
  {
    Real m = 0, V = 0, j = 0, u = 0, v = 0, a = 0, x = 0, y = 0, X = 0, Y = 0;

    Integrals(Real _m, Real _V, Real _j, Real _u, Real _v, Real _a,
      Real _x, Real _y, Real _X, Real _Y) :
    m(_m), V(_V), j(_j), u(_u), v(_v), a(_a), x(_x), y(_y), X(_X), Y(_Y) {}

    Integrals(const Integrals&c) :
    m(c.m),V(c.V),j(c.j),u(c.u),v(c.v),a(c.a),x(c.x),y(c.y),X(c.X),Y(c.Y){}

    void dimensionalize(const Real h, const Real cmx, const Real cmy)
    {
      //const Real eps = std::numeric_limits<Real>::epsilon();
      //first compute the center of mass:
      x /= m; y /= m;
      X /= V; Y /= V;
      //update the angular moment and moment of inertia:
      const Real dC[2] = {x-cmx, y-cmy};
      a += dC[0]*v -dC[1]*u;
      j += m*(dC[0]*dC[0] + dC[1]*dC[1]);
      //divide by area to get velocities
      u /= m; v /= m; a /= j;
      // multiply by grid-area to get proper mass , moment , and volume
      m *= std::pow(h, 2); j *= std::pow(h, 2); V *= std::pow(h, 2);
    }
  };

  Integrals integrateObstBlock(const vector<BlockInfo>& vInfo)
  {
    Real _m=0, _V=0, _j=0, _u=0, _v=0, _a=0, _x=0, _y=0, _X=0, _Y=0;
    #pragma omp parallel for schedule(dynamic) reduction( + : _m, _V, _j, _u, _v, _a, _x, _y, _X, _Y )
    for(size_t i=0; i<vInfo.size(); i++)
    {
      const auto pos = obstacleBlocks.find(vInfo[i].blockID);
      if(pos == obstacleBlocks.end()) continue;

      for(int iy=0; iy<FluidBlock::sizeY; ++iy)
      for(int ix=0; ix<FluidBlock::sizeX; ++ix)
      {
        const Real chi = pos->second->chi[iy][ix];
        if (chi <= 0) continue;
        Real p[2];
        vInfo[i].pos(p, ix, iy);
        const Real u_ = pos->second->udef[iy][ix][0];
        const Real v_ = pos->second->udef[iy][ix][1];
        const Real rhochi = chi*pos->second->rho[iy][ix];
        _x += rhochi*p[0];
        _y += rhochi*p[1];
        _X += chi*p[0];
        _Y += chi*p[1];
        p[0] -= centerOfMass[0];
        p[1] -= centerOfMass[1];
        _V += chi;
        _m += rhochi;
        _u += rhochi*u_;
        _v += rhochi*v_;
        _a += rhochi*(p[0]*v_ - p[1]*u_);
        _j += rhochi*(p[0]*p[0] + p[1]*p[1]);
      }
    }
    Integrals I(_m, _V, _j, _u, _v, _a, _x, _y, _X, _Y);
    I.dimensionalize(vInfo[0].h_gridpoint, centerOfMass[0], centerOfMass[1]);
    return I;
  }

  void removeMoments(const vector<BlockInfo>& vInfo)
  {
    Integrals I = integrateObstBlock(vInfo);
    #ifndef RL_TRAIN
    if(sim.verbose)
    if(fabs(I.u)+fabs(I.v)+fabs(I.a)>numeric_limits<Real>::epsilon())
      printf("Correction of: lin mom [%f %f] ang mom [%f]. Error in CM=[%f %f]\n", I.u, I.v, I.a, I.x-centerOfMass[0], I.y-centerOfMass[1]);
    #endif
    //update the center of mass, this operation should not move 'center'
    //cout << centerOfMass[0]<<" "<<I.x<<" "<<centerOfMass[1]<<" "<<I.y << endl;
    //cout << center[0]<<" "<<I.X<<" "<<center[1]<<" "<<I.Y << endl;
    centerOfMass[0] = I.x; centerOfMass[1] = I.y;
    //center[0] = I.X; center[1] = I.Y;
    const Real dCx = center[0]-centerOfMass[0];
    const Real dCy = center[1]-centerOfMass[1];
    d_gm[0] =  dCx*std::cos(orientation) +dCy*std::sin(orientation);
    d_gm[1] = -dCx*std::sin(orientation) +dCy*std::cos(orientation);

    #ifndef NDEBUG
      Real Cxtest = center[0] -std::cos(orientation)*d_gm[0] + std::sin(orientation)*d_gm[1];
      Real Cytest = center[1] -std::sin(orientation)*d_gm[0] - std::cos(orientation)*d_gm[1];
      if( std::abs(Cxtest-centerOfMass[0])>numeric_limits<Real>::epsilon() ||
          std::abs(Cytest-centerOfMass[1])>numeric_limits<Real>::epsilon() ) {
          printf("Error update of center of mass = [%f %f]\n", Cxtest-centerOfMass[0], Cytest-centerOfMass[1]);
            abort();
          }
    #endif

    #pragma omp parallel for schedule(dynamic)
    for(size_t i=0; i<vInfo.size(); i++)
    {
      const auto pos = obstacleBlocks.find(vInfo[i].blockID);
      if(pos == obstacleBlocks.end()) continue;

      for(int iy=0; iy<FluidBlock::sizeY; ++iy)
      for(int ix=0; ix<FluidBlock::sizeX; ++ix) {
          Real p[2];
          vInfo[i].pos(p, ix, iy);
          p[0] -= centerOfMass[0];
          p[1] -= centerOfMass[1];
          pos->second->udef[iy][ix][0] -= I.u -I.a*p[1];
          pos->second->udef[iy][ix][1] -= I.v +I.a*p[0];
      }
    }

    #ifndef NDEBUG
      Integrals Itest = integrateObstBlock(vInfo);
      if(std::abs(Itest.u)>numeric_limits<Real>::epsilon() ||
         std::abs(Itest.v)>numeric_limits<Real>::epsilon() ||
         std::abs(Itest.a)>numeric_limits<Real>::epsilon() ||
         std::abs(Itest.x-centerOfMass[0])>numeric_limits<Real>::epsilon() ||
         std::abs(Itest.y-centerOfMass[1])>numeric_limits<Real>::epsilon() ) {
       printf("After correction: linm [%f %f] angm [%f] deltaCM=[%f %f]\n",
       Itest.u,Itest.v,Itest.a,Itest.x-centerOfMass[0],Itest.y-centerOfMass[1]);
       abort();
      }
    #endif
  };

  void computeVelocities()
  {
  	const vector<BlockInfo>& vInfo = sim.grid->getBlocksInfo();
    Real _M = 0, _V = 0, _J = 0, um = 0, vm = 0, am = 0; //linear momenta
    #pragma omp parallel for schedule(dynamic) reduction(+:_M,_V,_J,um,vm,am)
    for(size_t i=0; i<vInfo.size(); i++) {
        FluidBlock & b = *(FluidBlock*)vInfo[i].ptrBlock;
        const auto pos = obstacleBlocks.find(vInfo[i].blockID);
        if(pos == obstacleBlocks.end()) continue;

        for(int iy=0; iy<FluidBlock::sizeY; ++iy)
        for(int ix=0; ix<FluidBlock::sizeX; ++ix) {
          const Real chi = pos->second->chi[iy][ix];
          if (chi <= 0) continue;
          Real p[2];
          vInfo[i].pos(p, ix, iy);
          p[0] -= centerOfMass[0];
          p[1] -= centerOfMass[1];

          const Real rhochi = pos->second->rho[iy][ix] * chi;
          _M += rhochi; _V += chi;
          um += rhochi * b(ix,iy).u;
          vm += rhochi * b(ix,iy).v;
          _J += rhochi * (p[0]*p[0]       + p[1]*p[1]);
          am += rhochi * (p[0]*b(ix,iy).v - p[1]*b(ix,iy).u);
        }
    }

    computedu = um / _M;
    computedv = vm / _M;
    computedo = am / _J;
    if(bForcedx) u = forcedu;
    else         u = computedu;

    if(bForcedy) v = forcedv;
    else         v = computedv;

    if(not bBlockang) omega = computedo;

    J = _J * std::pow(vInfo[0].h_gridpoint,2);
    M = _M * std::pow(vInfo[0].h_gridpoint,2);
    V = _V * std::pow(vInfo[0].h_gridpoint,2);

    #ifndef RL_TRAIN
      if(sim.verbose)
        printf("CM:[%f %f] C:[%f %f] u:%f v:%f omega:%f M:%f J:%f V:%f\n",
        centerOfMass[0], centerOfMass[1], center[0], center[1], u, v, omega, M, J, V);

      {
        ofstream fileSpeed;
      	stringstream ssF;
      	ssF<<"velocity_0.dat";

        fileSpeed.open(ssF.str().c_str(), ios::app);
        if(sim.step==0)
          fileSpeed<<"time dt CMx CMy angle u v omega M J accx accy"<<std::endl;

        fileSpeed<<sim.time<<" "<<sim.dt<<" "<<centerOfMass[0]<<" "<<centerOfMass[1]<<" "<<orientation<<" "<<u <<" "<<v<<" "<<omega <<" "<<M<<" "<<J<<endl;
        fileSpeed.close();
      }
    #endif
  }

  void updateLabVelocity( int nSum[2], double uSum[2] )
  {
    if(bFixedx) { (nSum[0])++; uSum[0] -= u; }
    if(bFixedy) { (nSum[1])++; uSum[1] -= v; }
  }

  void penalize()
  {
  	const vector<BlockInfo>& vInfo = sim.grid->getBlocksInfo();
    const Real lamdt = sim.dt * sim.lambda;
    #pragma omp parallel for schedule(dynamic)
    for(size_t i=0; i<vInfo.size(); i++) {
      FluidBlock & b = *(FluidBlock*)vInfo[i].ptrBlock;
      const auto pos = obstacleBlocks.find(vInfo[i].blockID);
      if(pos == obstacleBlocks.end()) continue;

      for(int iy=0; iy<FluidBlock::sizeY; ++iy)
      for(int ix=0; ix<FluidBlock::sizeX; ++ix) {
        const Real chi = pos->second->chi[iy][ix];
        if (chi <= 0) continue;
        Real p[2];
        vInfo[i].pos(p, ix, iy);
        p[0] -= centerOfMass[0]; p[1] -= centerOfMass[1];
        const Real alpha = 1/(1 + lamdt * chi);
        const Real* udef = pos->second->udef[iy][ix];
        const Real uTot = u -omega*p[1] +udef[0];// -sim.uinfx;
        const Real vTot = v +omega*p[0] +udef[1];// -sim.uinfy;
        b(ix,iy).u = alpha*b(ix,iy).u + (1-alpha)*uTot;
        b(ix,iy).v = alpha*b(ix,iy).v + (1-alpha)*vTot;
      }
    }
  }

  void diagnostics()
  {
    /*
    const vector<BlockInfo>& vInfo = sim.grid->getBlocksInfo();
    Real cX=0, cY=0, cmX=0, cmY=0, fx=0, fy=0, pMin=10, pMax=0, mass=0, volS=0, volF=0;
    const double dh = sim.getH();

    #pragma omp parallel for reduction(max : pMax) reduction (min : pMin) reduction(+ : cX,cY,volF,cmX,cmY,fx,fy,mass,volS)
    for(int i=0; i<vInfo.size(); i++) {
      FluidBlock& b = *(FluidBlock*)vInfo[i].ptrBlock;
      const auto pos = obstacleBlocks.find(vInfo[i].blockID);

      for(int iy=0; iy<FluidBlock::sizeY; ++iy)
      for(int ix=0; ix<FluidBlock::sizeX; ++ix) {
        double p[2] = {0,0};
        vInfo[i].pos(p, ix, iy);
        const Real chi = pos==obstacleBlocks.end()?0 : pos->second->chi[iy][ix];
        const Real rhochi = pos->second->rho[iy][ix] * chi;
        cmX += p[0] * rhochi; cX += p[0] * chi; fx += (b(ix,iy).u-uBody)*chi;
        cmY += p[1] * rhochi; cY += p[1] * chi; fy += (b(ix,iy).v-vBody)*chi;
        mass += rhochi; volS += chi; volF += (1-chi);
        pMin = min(pMin,(double)b(ix,iy).p);
        pMax = max(pMax,(double)b(ix,iy).p);
      }
    }

    cmX /= mass; cX /= volS; fx *= dh*dh*lambda;
    cmY /= mass; cY /= volS; fy *= dh*dh*lambda;
    const Real rhoSAvg = mass/volS, length = getCharLength();
    const Real speed = std::sqrt ( uBody * uBody + vBody * vBody);
    const Real cDx = 2*fx/(speed*speed*length);
    const Real cDy = 2*fy/(speed*speed*length);
    const Real Re_uBody = getCharLength()*speed/nu;
    const Real theta = getOrientation();
    volS *= dh*dh; volF *= dh*dh;
    Real CO[2], CM[2], labpos[2];
    getLabPosition(labpos);
    getCentroid(CO);
    getCenterOfMass(CM);
    stringstream ss;
    ss << "./diagnostics_0.dat";
    ofstream myfile(ss.str(), fstream::app);
    if (!step)
    myfile<<"step time CO[0] CO[1] CM[0] CM[1] centroidX centroidY centerMassX centerMassY labpos[0] labpos[1] theta uBody[0] uBody[1] omegaBody Re_uBody cDx cDy rhoSAvg"<<endl;

    cout<<step<<" "<<time<<" "<<CO[0]<<" "<<CO[1]<<" "<<CM[0]<<" "<<CM[1]
      <<" " <<cX<<" "<<cY<<" "<<cmX<<" "<<cmY<<" "<<labpos[0]<<" "<<labpos[1]
      <<" "<<theta<<" "<<uBody<<" "<<vBody<<" "<<omegaBody<<" "<<Re_uBody
      <<" "<<cDx<<" "<<cDy<<" "<<rhoSAvg<<" "<<fx<<" "<<fy<<endl;

    myfile<<step<<" "<<time<<" "<<CO[0]<<" "<<CO[1]<<" "<<CM[0]<<" "<<CM[1]
      <<" " <<cX<<" "<<cY<<" "<<cmX<<" "<<cmY<<" "<<labpos[0]<<" "<<labpos[1]
      <<" "<<theta<<" "<<uBody<<" "<<vBody<<" "<<omegaBody<<" "<<Re_uBody
      <<" "<<cDx<<" "<<cDy<<" "<<rhoSAvg<<" "<<fx<<" "<<fy<<endl;

    myfile.close();
    */
  }

  void characteristic_function()
  {
  	const vector<BlockInfo>& vInfo = sim.grid->getBlocksInfo();
    #pragma omp parallel for schedule(dynamic)
    for(size_t i=0; i<vInfo.size(); i++) {
      FluidBlock& b = *(FluidBlock*)vInfo[i].ptrBlock;
      const auto pos = obstacleBlocks.find(vInfo[i].blockID);
      if(pos == obstacleBlocks.end())
        for(int iy=0; iy<FluidBlock::sizeY; ++iy)
        for(int ix=0; ix<FluidBlock::sizeX; ++ix)
          b(ix,iy).tmp = 0;
      else
        for(int iy=0; iy<FluidBlock::sizeY; ++iy)
        for(int ix=0; ix<FluidBlock::sizeX; ++ix)
          b(ix,iy).tmp = pos->second->chi[iy][ix];
    }
  }

  template <typename Kernel>
  void compute(const Kernel& kernel, const vector<BlockInfo>& vInfo)
  {
    #pragma omp parallel
    {
      Lab mylab;
      mylab.prepare(*(sim.grid), kernel.stencil_start, kernel.stencil_end, false);

      #pragma omp for schedule(dynamic)
      for (size_t i=0; i<vInfo.size(); i++)
      {
        const auto pos = obstacleBlocks.find(vInfo[i].blockID);
        if(pos == obstacleBlocks.end()) continue;
        mylab.load(vInfo[i], 0);
        FluidBlock& b = *(FluidBlock*)vInfo[i].ptrBlock;
        kernel(mylab, vInfo[i], b, pos->second);
      }
    }
  }

  template <typename Kernel>
  void compute_surface(const Kernel& kernel, const vector<BlockInfo>& vInfo)
  {
    #pragma omp parallel
    {
      Lab mylab;
      mylab.prepare(*(sim.grid), kernel.stencil_start, kernel.stencil_end, false);

      #pragma omp for schedule(dynamic)
      for (size_t i=0; i<vInfo.size(); i++)
      {
        const auto pos = obstacleBlocks.find(vInfo[i].blockID);
        if(pos == obstacleBlocks.end()) continue; //obstacle is not in the block
        assert(pos->second->filled);
        if(!pos->second->n_surfPoints) continue; //does not contain surf points

        mylab.load(vInfo[i], 0);
        FluidBlock& b = *(FluidBlock*)vInfo[i].ptrBlock;
        kernel(mylab, vInfo[i], b, pos->second);
      }
    }
  }

  void computeForces()
  {
  	const vector<BlockInfo>& vInfo = sim.grid->getBlocksInfo();
    Real vel_unit[2] = {0., 0.};
    const Real vel_norm = std::sqrt(u*u + v*v);
    if(vel_norm>0){ vel_unit[0]=u/vel_norm; vel_unit[1]=v/vel_norm; }

    OperatorComputeForces finalize(sim.nu, vel_unit, centerOfMass);
    compute_surface(finalize, vInfo);

    //additive quantities:
    Real perimeter = 0, forcex = 0, forcey = 0, forcex_P = 0, forcey_P = 0, forcex_V = 0, forcey_V = 0, torque = 0, torque_P = 0, torque_V = 0, drag = 0, thrust = 0, Pout = 0, PoutBnd = 0, defPower = 0, defPowerBnd = 0, circulation = 0;

    for (auto & block : obstacleBlocks)
    {
      circulation += block.second->circulation;
      perimeter   += block.second->perimeter;
      forcex      += block.second->forcex;
      forcey      += block.second->forcey;
      forcex_P    += block.second->forcex_P;
      forcey_P    += block.second->forcey_P;
      forcex_V    += block.second->forcex_V;
      forcey_V    += block.second->forcey_V;
      torque      += block.second->torque;
      torque_P    += block.second->torque_P;
      torque_V    += block.second->torque_V;
      drag        += block.second->drag;
      thrust      += block.second->thrust;
      Pout        += block.second->Pout;
      PoutBnd     += block.second->PoutBnd;
      defPower    += block.second->defPower;
      defPowerBnd += block.second->defPowerBnd;
    }

    //derived quantities:
    Real Pthrust    = thrust*vel_norm;
    Real Pdrag      =   drag*vel_norm;
    Real EffPDef    = Pthrust/(Pthrust-min(defPower,(Real)0.));
    Real EffPDefBnd = Pthrust/(Pthrust-    defPowerBnd);

    #ifndef RL_TRAIN
    if (sim._bDump) {
      char buf[500];
      sprintf(buf, "surface_0_%07d.raw", sim.step);
      FILE * pFile = fopen (buf, "wb");
      for(auto & block : obstacleBlocks) block.second->print(pFile);
      fflush(pFile);
      fclose(pFile);
    }

    {
      ofstream fileForce;
      ofstream filePower;
    	stringstream ssF, ssP;
    	ssF<<"forceValues_0.dat";
    	ssP<<"powerValues_0.dat"; //obstacleID

      fileForce.open(ssF.str().c_str(), ios::app);
      if(sim.step==0)
        fileForce<<"time Fx Fy FxPres FyPres FxVisc FyVisc tau tauPres tauVisc drag thrust perimeter circulation"<<std::endl;

      fileForce<<sim.time<<" "<<forcex<<" "<<forcey<<" "<<forcex_P<<" "<<forcey_P<<" "<<forcex_V <<" "<<forcey_V<<" "<<torque <<" "<<torque_P<<" "<<torque_V<<" "<<drag<<" "<<thrust<<" "<<perimeter<<" "<<circulation<<endl;
      fileForce.close();

      filePower.open(ssP.str().c_str(), ios::app);
      if(sim.step==0)
        filePower<<"time Pthrust Pdrag PoutBnd Pout defPowerBnd defPower EffPDefBnd EffPDef"<<std::endl;
      filePower<<sim.time<<" "<<Pthrust<<" "<<Pdrag<<" "<<PoutBnd<<" "<<Pout<<" "<<defPowerBnd<<" "<<defPower<<" "<<EffPDefBnd<<" "<<EffPDef<<endl;
      filePower.close();
    }
    #endif
  }
};
