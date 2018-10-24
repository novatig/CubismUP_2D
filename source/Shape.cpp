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

#include "Shape.h"
#include "OperatorComputeForces.h"
#include "BufferedLogger.h"

Real Shape::getMinRhoS() const { return rhoS; }
bool Shape::bVariableDensity() const {
  return std::fabs(rhoS-1.0) > numeric_limits<Real>::epsilon();
}

void Shape::updatePosition(double dt)
{
  // update centerOfMass - this is the reference point from which we compute the center
  // Remember, uinf is -ubox, therefore we sum it to u body to get
  // velocity of shapre relative to the sim box
  centerOfMass[0] += dt*( u + sim.uinfx );
  centerOfMass[1] += dt*( v + sim.uinfy );
  labCenterOfMass[0] += dt*u;
  labCenterOfMass[1] += dt*v;

  orientation += dt*omega;
  orientation = orientation> M_PI ? orientation-2*M_PI : orientation;
  orientation = orientation<-M_PI ? orientation+2*M_PI : orientation;

  center[0] = centerOfMass[0] + std::cos(orientation)*d_gm[0] - std::sin(orientation)*d_gm[1];
  center[1] = centerOfMass[1] + std::sin(orientation)*d_gm[0] + std::cos(orientation)*d_gm[1];
}

void Shape::outputSettings(ostream &outStream) const
{
  outStream << "centerX " << center[0] << endl;
  outStream << "centerY " << center[1] << endl;
  outStream << "centerMassX " << centerOfMass[0] << endl;
  outStream << "centerMassY " << centerOfMass[1] << endl;
  outStream << "orientation " << orientation << endl;
  outStream << "rhoS " << rhoS << endl;
}

Shape::Integrals Shape::integrateObstBlock(const vector<BlockInfo>& vInfo)
{
  double _x=0, _y=0, _m=0, _j=0, _u=0, _v=0, _a=0;
  const double hsq = std::pow(vInfo[0].h_gridpoint, 2);
  #pragma omp parallel for schedule(dynamic) reduction(+: _x,_y,_m,_j,_u,_v,_a)
  for(size_t i=0; i<vInfo.size(); i++) {
    const auto pos = obstacleBlocks.find(vInfo[i].blockID);
    if(pos == obstacleBlocks.end()) continue;

    for(int iy=0; iy<FluidBlock::sizeY; ++iy)
    for(int ix=0; ix<FluidBlock::sizeX; ++ix) {
      const Real chi = pos->second->chi[iy][ix];
      if (chi <= 0) continue;
      double p[2];
      vInfo[i].pos(p, ix, iy);
      const double u_ = pos->second->udef[iy][ix][0];
      const double v_ = pos->second->udef[iy][ix][1];
      const double rhochi = chi * pos->second->rho[iy][ix] * hsq;
      _x += rhochi*p[0];
      _y += rhochi*p[1];
      p[0] -= centerOfMass[0];
      p[1] -= centerOfMass[1];
      _m += rhochi;
      _j += rhochi*(p[0]*p[0] + p[1]*p[1]);
      _u += rhochi*u_;
      _v += rhochi*v_;
      _a += rhochi*(p[0]*v_ - p[1]*u_);
    }
  }
  _x /= _m; _y /= _m;
  // Parallel axis theorem:
  const double dC[2] = { _x - centerOfMass[0], _y - centerOfMass[1] };
  // I_arbitrary_axis = I_CM + m * dist_CM_axis ^ 2 . Now _j is J around old CM
  _j = _j - _m*(dC[0]*dC[0] + dC[1]*dC[1]);
  _a = _a - ( dC[0]*_v - dC[1]*_u );
  // turn moments into velocities:
  _u /= _m;  _v /= _m;  _a /= _j;
  return Integrals(_x, _y, _m, _j, _u, _v, _a);
}

void Shape::removeMoments(const vector<BlockInfo>& vInfo)
{
  Shape::Integrals I = integrateObstBlock(vInfo);

  #ifndef RL_TRAIN
    if(sim.verbose)
    if(fabs(I.u)+fabs(I.v)+fabs(I.a)>10*numeric_limits<Real>::epsilon())
      printf("Correct: lin mom [%f %f] ang mom [%f]. Error in CM=[%f %f]\n",
        I.u, I.v, I.a, I.x-centerOfMass[0], I.y-centerOfMass[1]);
  #endif
  //update the center of mass, this operation should not move 'center'
  //cout << centerOfMass[0]<<" "<<I.x<<" "<<centerOfMass[1]<<" "<<I.y << endl;
  //cout << center[0]<<" "<<I.X<<" "<<center[1]<<" "<<I.Y << endl;
  centerOfMass[0] = I.x; centerOfMass[1] = I.y;
  //center[0] = I.X; center[1] = I.Y;
  const double dCx = center[0]-centerOfMass[0];
  const double dCy = center[1]-centerOfMass[1];
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
  for(size_t i=0; i<vInfo.size(); i++) {
    const auto pos = obstacleBlocks.find(vInfo[i].blockID);
    if(pos == obstacleBlocks.end()) continue;

    for(int iy=0; iy<FluidBlock::sizeY; ++iy)
    for(int ix=0; ix<FluidBlock::sizeX; ++ix) {
        double p[2];
        vInfo[i].pos(p, ix, iy);
        p[0] -= centerOfMass[0];
        p[1] -= centerOfMass[1];
        pos->second->udef[iy][ix][0] -= I.u -I.a*p[1];
        pos->second->udef[iy][ix][1] -= I.v +I.a*p[0];
    }
  }

  #ifndef NDEBUG
   Shape::Integrals Itest = integrateObstBlock(vInfo);
   if(std::abs(Itest.u)>10*numeric_limits<Real>::epsilon() ||
      std::abs(Itest.v)>10*numeric_limits<Real>::epsilon() ||
      std::abs(Itest.a)>10*numeric_limits<Real>::epsilon() ||
      std::abs(Itest.x-centerOfMass[0])>10*numeric_limits<Real>::epsilon() ||
      std::abs(Itest.y-centerOfMass[1])>10*numeric_limits<Real>::epsilon() ){
    printf("After correction: linm [%e %e] angm [%e] deltaCM=[%e %e]\n",
    Itest.u,Itest.v,Itest.a,Itest.x-centerOfMass[0],Itest.y-centerOfMass[1]);
    abort();
   }
  #endif
};

void Shape::computeVelocities()
{
  const vector<BlockInfo>& vInfo = sim.grid->getBlocksInfo();
  double _M = 0, _J = 0, um = 0, vm = 0, am = 0; //linear momenta
  const double hsq = std::pow(vInfo[0].h_gridpoint,2);
  #pragma omp parallel for schedule(dynamic) reduction(+:_M,_J,um,vm,am)
  for(size_t i=0; i<vInfo.size(); i++) {
      FluidBlock & b = *(FluidBlock*)vInfo[i].ptrBlock;
      const auto pos = obstacleBlocks.find(vInfo[i].blockID);
      if(pos == obstacleBlocks.end()) continue;

      for(int iy=0; iy<FluidBlock::sizeY; ++iy)
      for(int ix=0; ix<FluidBlock::sizeX; ++ix) {
        const Real chi = pos->second->chi[iy][ix];
        if (chi <= 0) continue;
        double p[2];
        vInfo[i].pos(p, ix, iy);
        p[0] -= centerOfMass[0];
        p[1] -= centerOfMass[1];

        const double rhochi = pos->second->rho[iy][ix] * chi * hsq;
        _M += rhochi;
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

  #ifndef RL_TRAIN
    if(sim.verbose)
      printf("CM:[%f %f] C:[%f %f] u:%f v:%f omega:%f M:%f J:%f V:%f\n",
      centerOfMass[0], centerOfMass[1], center[0], center[1], u, v, omega, M, J, V);
    if(not sim.muteAll)
    {
      stringstream ssF;
      ssF<<sim.path2file<<"/velocity_"<<obstacleID<<".dat";
      std::stringstream &fileSpeed = logger.get_stream(ssF.str());
      if(sim.step==0)
        fileSpeed<<"time dt CMx CMy angle u v omega M J accx accy"<<std::endl;

      fileSpeed<<sim.time<<" "<<sim.dt<<" "<<centerOfMass[0]<<" "<<centerOfMass[1]<<" "<<orientation<<" "<<u <<" "<<v<<" "<<omega <<" "<<M<<" "<<J<<endl;
    }
  #endif
}

void Shape::updateLabVelocity( int nSum[2], double uSum[2] )
{
  if(bFixedx) { (nSum[0])++; uSum[0] -= u; }
  if(bFixedy) { (nSum[1])++; uSum[1] -= v; }
}

void Shape::penalize()
{
  const vector<BlockInfo>& vInfo = sim.grid->getBlocksInfo();
  const double lamdt = sim.dt * sim.lambda;
  #pragma omp parallel for schedule(dynamic)
  for(size_t i=0; i<vInfo.size(); i++) {
    FluidBlock & b = *(FluidBlock*)vInfo[i].ptrBlock;
    const auto pos = obstacleBlocks.find(vInfo[i].blockID);
    if(pos == obstacleBlocks.end()) continue;

    for(int iy=0; iy<FluidBlock::sizeY; ++iy)
    for(int ix=0; ix<FluidBlock::sizeX; ++ix) {
      const Real chi = pos->second->chi[iy][ix];
      if (chi <= 0) continue;
      double p[2];
      vInfo[i].pos(p, ix, iy);
      p[0] -= centerOfMass[0]; p[1] -= centerOfMass[1];
      const double alpha = 1/(1 + lamdt * chi);
      const Real*const udef = pos->second->udef[iy][ix];
      const double uTot = u -omega*p[1] +udef[0];// -sim.uinfx;
      const double vTot = v +omega*p[0] +udef[1];// -sim.uinfy;
      b(ix,iy).u = alpha*b(ix,iy).u + (1-alpha)*uTot;
      b(ix,iy).v = alpha*b(ix,iy).v + (1-alpha)*vTot;
    }
  }
}

void Shape::diagnostics()
{
  const vector<BlockInfo>& vInfo = sim.grid->getBlocksInfo();
  const double hsq = std::pow(vInfo[0].h_gridpoint, 2);
  double _a=0, _m=0, _x=0, _y=0, _t=0;
  #pragma omp parallel for schedule(dynamic) reduction(+:_a,_m,_x,_y,_t)
  for(size_t i=0; i<vInfo.size(); i++) {
      const auto pos = obstacleBlocks.find(vInfo[i].blockID);
      if(pos == obstacleBlocks.end()) continue;
      FluidBlock& b = *(FluidBlock*)vInfo[i].ptrBlock;

      for(int iy=0; iy<FluidBlock::sizeY; ++iy)
      for(int ix=0; ix<FluidBlock::sizeX; ++ix) {
        if (pos->second->chi[iy][ix] <= 0) continue;
        const double Xs = pos->second->chi[iy][ix] * hsq;
        double p[2];
        vInfo[i].pos(p, ix, iy);
        p[0] -= centerOfMass[0];
        p[1] -= centerOfMass[1];
        const Real*const udef = pos->second->udef[iy][ix];
        const double uDiff = b(ix,iy).u - (u -omega*p[1] +udef[0]);
        const double vDiff = b(ix,iy).v - (v +omega*p[0] +udef[1]);
        _a += Xs;
        _m += Xs / b(ix,iy).invRho;
        _x += uDiff*Xs;
        _y += vDiff*Xs;
        _t += (p[0]*vDiff-p[1]*uDiff)*Xs;
      }
  }
  area_penal   = _a;
  mass_penal   = _m;
  forcex_penal = _x * sim.lambda;
  forcey_penal = _y * sim.lambda;
  torque_penal = _t * sim.lambda;
}

void Shape::characteristic_function()
{
  const vector<BlockInfo>& vInfo = sim.grid->getBlocksInfo();
  #pragma omp parallel for schedule(dynamic)
  for(size_t i=0; i<vInfo.size(); i++) {
    FluidBlock& b = *(FluidBlock*)vInfo[i].ptrBlock;
    const auto pos = obstacleBlocks.find(vInfo[i].blockID);
    if(pos == obstacleBlocks.end()) continue;

    for(int iy=0; iy<FluidBlock::sizeY; ++iy)
    for(int ix=0; ix<FluidBlock::sizeX; ++ix)
      b(ix,iy).tmp = std::max(pos->second->chi[iy][ix], b(ix,iy).tmp);
  }
}

void Shape::deformation_velocities()
{
  const vector<BlockInfo>& vInfo = sim.grid->getBlocksInfo();
  #pragma omp parallel for schedule(dynamic)
  for(size_t i=0; i<vInfo.size(); i++) {
    FluidBlock& b = *(FluidBlock*)vInfo[i].ptrBlock;
    const auto pos = obstacleBlocks.find(vInfo[i].blockID);
    if(pos == obstacleBlocks.end()) continue;

    for(int iy=0; iy<FluidBlock::sizeY; ++iy)
    for(int ix=0; ix<FluidBlock::sizeX; ++ix) {
      b(ix,iy).tmpU += pos->second->udef[iy][ix][0];
      b(ix,iy).tmpV += pos->second->udef[iy][ix][1];
    }
  }
}

void Shape::computeForces()
{
  const vector<BlockInfo>& vInfo = sim.grid->getBlocksInfo();
  Real vel_unit[2] = {0., 0.};
  const Real vel_norm = std::sqrt(u*u + v*v);
  if(vel_norm>0){ vel_unit[0]=u/vel_norm; vel_unit[1]=v/vel_norm; }

  OperatorComputeForces finalize(sim.nu, vel_unit, centerOfMass);
  compute_surface(finalize, vInfo);

  //additive quantities:
  perimeter = 0; forcex = 0; forcey = 0; forcex_P = 0;
  forcey_P = 0; forcex_V = 0; forcey_V = 0; torque = 0;
  torque_P = 0; torque_V = 0; drag = 0; thrust = 0;
  Pout = 0; PoutBnd = 0; defPower = 0; defPowerBnd = 0; circulation = 0;

  for (auto & block : obstacleBlocks)
  {
    circulation += block.second->circulation;
    perimeter += block.second->perimeter; torque   += block.second->torque;
    forcex   += block.second->forcex;     forcey   += block.second->forcey;
    forcex_P += block.second->forcex_P;   forcey_P += block.second->forcey_P;
    forcex_V += block.second->forcex_V;   forcey_V += block.second->forcey_V;
    torque_P += block.second->torque_P;   torque_V += block.second->torque_V;
    drag     += block.second->drag;       thrust   += block.second->thrust;
    Pout += block.second->Pout; defPowerBnd += block.second->defPowerBnd;
    PoutBnd += block.second->PoutBnd; defPower += block.second->defPower;
  }

  //derived quantities:
  Pthrust    = thrust*vel_norm;
  Pdrag      =   drag*vel_norm;
  EffPDef    = Pthrust/(Pthrust-min(defPower,(double)0));
  EffPDefBnd = Pthrust/(Pthrust-    defPowerBnd);

  #ifndef RL_TRAIN
  if (sim._bDump && not sim.muteAll)
  {
    stringstream ssF; ssF<<sim.path2file<<"/surface_"<<obstacleID
      <<"_"<<std::setfill('0')<<std::setw(7)<<sim.step<<".raw";
    ofstream pFile(ssF.str().c_str(), ofstream::binary);
    for(auto & block : obstacleBlocks) block.second->print(pFile);
    pFile.close();
  }
  if(not sim.muteAll)
  {
    stringstream ssF, ssP;
    ssF<<sim.path2file<<"/forceValues_"<<obstacleID<<".dat";
    ssP<<sim.path2file<<"/powerValues_"<<obstacleID<<".dat"; //obstacleID

    std::stringstream &fileForce = logger.get_stream(ssF.str());
    if(sim.step==0)
      fileForce<<"time Fx Fy FxPres FyPres FxVisc FyVisc tau tauPres tauVisc drag thrust perimeter circulation area_penal mass_penal forcex_penal forcey_penal torque_penal"<<std::endl;

    fileForce<<sim.time<<" "<<forcex<<" "<<forcey<<" "<<forcex_P<<" "<<forcey_P<<" "<<forcex_V <<" "<<forcey_V<<" "<<torque <<" "<<torque_P<<" "<<torque_V<<" "<<drag<<" "<<thrust<<" "<<perimeter<<" "<<circulation<<" "<<area_penal<<" "<<mass_penal<<" "<<forcex_penal<<" "<<forcey_penal<<" "<<torque_penal<<endl;

    std::stringstream &filePower = logger.get_stream(ssP.str());
    if(sim.step==0)
      filePower<<"time Pthrust Pdrag PoutBnd Pout defPowerBnd defPower EffPDefBnd EffPDef"<<std::endl;
    filePower<<sim.time<<" "<<Pthrust<<" "<<Pdrag<<" "<<PoutBnd<<" "<<Pout<<" "<<defPowerBnd<<" "<<defPower<<" "<<EffPDefBnd<<" "<<EffPDef<<endl;
  }
  #endif
}

